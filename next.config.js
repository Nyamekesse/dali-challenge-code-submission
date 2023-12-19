/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_ENVIRONMENT: process.env.PINECONE_ENVIRONMENT,
    PINECONE_INDEX: process.env.PINECONE_INDEX,
    SERPAPI_API_KEY: process.env.SERPAPI_API_KEY,
  },
};

module.exports = nextConfig;
