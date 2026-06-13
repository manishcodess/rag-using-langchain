import * as dotenv from 'dotenv';
import fs from 'node:fs';
import path from 'node:path';
import readlineSync from 'readline-sync';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

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

let ai;
const History = [];

function validateEnv() {
  const required = ['GEMINI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME'];
  const missing = required.filter((name) => !process.env[name]);

  if (missing.length > 0) {
    throw new Error(`Missing required environment variable(s): ${missing.join(', ')}`);
  }
}


async function transformQuery(question){

History.push({
    role:'user',
    parts:[{text:question}]
    })  

const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    Only output the rewritten question and nothing else.
      `,
    },
 });
 
 History.pop()
 
 return response.text;

}


async function chatting(question) {

    // covert this question into vector
    
    const queries = await transformQuery(question);

 const embedResponse = await ai.models.embedContent({
   model: 'gemini-embedding-001',
   contents: queries,
   config: { outputDimensionality: 768 },
 });
 const queryVector = embedResponse.embeddings?.[0]?.values ?? [];

 if (queryVector.length === 0) {
   throw new Error('Embedding model returned an empty vector for the query.');
 }
//  query vector


// make connection with pinecone
 const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
 const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
    });

//   console.log(searchResults);  

//   top 10 documents: 10 metadata text part 10 documebnt

const context = searchResults.matches
                   .map((match) => match?.metadata?.text)
                   .filter(Boolean)
                   .join("\n\n---\n\n");

if (!context) {
  console.log("\nI could not find relevant context in Pinecone. Try indexing first or ask a more specific question.\n");
  return;
}
// create the context for the LLM

// Gemini


History.push({
    role:'user',
    parts:[{text:queries}]
    })  


    const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: ${context}
      `,
    },
   });


  History.push({
    role:'model',
    parts:[{text:response.text}]
  });

  console.log("\n");
  console.log(response.text);

}


async function main(){
  validateEnv();
  ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  console.log('Ask questions about your indexed PDF. Type "exit" to quit.');

  while (true) {
    const userProblem = readlineSync.question('Ask me anything --> ').trim();

    if (!userProblem) {
      continue;
    }

    if (userProblem.toLowerCase() === 'exit') {
      console.log('Goodbye!');
      break;
    }

    try {
      await chatting(userProblem);
    } catch (error) {
      console.error('Query failed:', error.message);
    }
  }
}


main().catch((error) => {
  console.error('Application failed:', error.message);
  process.exit(1);
});