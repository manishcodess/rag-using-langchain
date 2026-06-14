import * as dotenv from 'dotenv';
import fs from 'node:fs';
import path from 'node:path';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenAI } from '@google/genai';
import { Pinecone } from '@pinecone-database/pinecone';

function loadEnvFile() {
  const envPath = path.resolve('.env');
  if (!fs.existsSync(envPath)) {
    return;
  }

  const parsed = dotenv.parse(fs.readFileSync(envPath, 'utf8'));
  for (const [key, value] of Object.entries(parsed)) {
    process.env[key] = String(value).trim();
  }
}

loadEnvFile();

function validateEnv() {
  const required = ['GEMINI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME'];
  const missing = required.filter((name) => !process.env[name]);

  if (missing.length > 0) {
    throw new Error(`Missing required environment variable(s): ${missing.join(', ')}`);
  }
}

async function embedWithRetry(ai, text, retries = 5, initialDelay = 1000) {
  let delay = initialDelay;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const embedResponse = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: text,
        config: { outputDimensionality: 768 },
      });
      const values = embedResponse.embeddings?.[0]?.values ?? [];
      if (values.length === 0) {
        throw new Error('Empty embedding vector returned');
      }
      return values;
    } catch (error) {
      if (attempt === retries) {
        throw error;
      }
      console.warn(`[WARNING] Embedding attempt ${attempt} failed: ${error.message}. Retrying in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay *= 2;
    }
  }
}


async function indexDocument() {
  validateEnv();

  const inputPdf = process.argv[2] ?? './dsa.pdf';
  const pdfPath = path.resolve(inputPdf);

  if (!fs.existsSync(pdfPath)) {
    throw new Error(`PDF file not found at: ${pdfPath}`);
  }

  const pdfLoader = new PDFLoader(pdfPath);
  const rawDocs = await pdfLoader.load();
  console.log('PDF loaded');

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  const cleanedDocs = chunkedDocs.filter((doc) => doc.pageContent?.trim().length > 0);

  if (cleanedDocs.length === 0) {
    throw new Error('No text could be extracted from the PDF. Use a text-based PDF (not scanned image-only).');
  }

  const maxChunks = Number(process.env.INDEX_MAX_CHUNKS || 0);
  const docsToIndex = Number.isFinite(maxChunks) && maxChunks > 0
    ? cleanedDocs.slice(0, maxChunks)
    : cleanedDocs;

  console.log(`Chunking completed (${docsToIndex.length}/${chunkedDocs.length} chunks selected)`);

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  console.log('Embedding model configured');

  const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
  console.log('Pinecone configured');

  const texts = docsToIndex.map((doc) => doc.pageContent);
  const vectors = [];
  for (let i = 0; i < texts.length; i += 1) {
    if (i > 0) {
      // Small delay to prevent rapid-fire requests triggering rate limits or 503s
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    const values = await embedWithRetry(ai, texts[i]);
    vectors.push(values);

    if ((i + 1) % 10 === 0 || i + 1 === texts.length) {
      console.log(`Embedded ${i + 1}/${texts.length} chunks`);
    }
  }

  if (vectors.length === 0) {
    throw new Error('Embedding model returned 0 vectors for the document chunks.');
  }

  const records = vectors.map((values, i) => {
    const doc = docsToIndex[i];
    const source = String(doc.metadata?.source ?? pdfPath);
    const page = String(doc.metadata?.loc?.pageNumber ?? doc.metadata?.pageNumber ?? 'unknown');

    return {
      id: `${path.basename(source)}-${page}-${i}-${Date.now()}`,
      values,
      metadata: {
        text: doc.pageContent,
        source,
        page,
      },
    };
  });

  const batchSize = 50;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    await pineconeIndex.upsert({ records: batch });
    console.log(`Upserted ${Math.min(i + batch.length, records.length)}/${records.length} vectors`);
  }

  console.log('Data stored successfully');
}

indexDocument().catch((error) => {
  console.error('Indexing failed:', error.message);
  process.exit(1);
});
