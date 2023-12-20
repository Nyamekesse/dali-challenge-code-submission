import { YoutubeTranscript } from "youtube-transcript";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";

let chain;

let chatHistory = [];

const initializeChain = async (initialPrompt, transcript) => {
  try {
    const model = new ChatOpenAI({
      temperature: 0.8,
      modelName: "gpt-3.5-turbo",
    });

    const splitter = new CharacterTextSplitter({
      separator: " ",
      chunkSize: 7,
      chunkOverlap: 3,
    });

    const docs = await splitter.createDocuments([transcript]);

    const vectorStore = await HNSWLib.fromDocuments(
      [{ pageContent: transcript }],
      new OpenAIEmbeddings()
    );

    const rootDir = process.cwd();

    await vectorStore.save(rootDir);

    const loadedVectorStore = await HNSWLib.load(
      rootDir,
      new OpenAIEmbeddings()
    );

    chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStore.asRetriever(),
      { verbose: true }
    );

    const response = await chain.call({
      question: initialPrompt,
      chat_history: chatHistory,
    });

    // Update history
    chatHistory.push({
      role: "assistant",
      content: response.text,
    });

    return response;
  } catch (error) {
    console.error(error);
  }
};

export default async function handler(req, res) {
  if (req.method === "POST") {
    const { prompt } = req.body;
    const { firstMsg } = req.body;

    if (firstMsg) {
      try {
        const initialPrompt = `Give me a summary of the transcript: ${prompt}`;

        chatHistory.push({
          role: "user",
          content: initialPrompt,
        });

        const transcriptResponse = await YoutubeTranscript.fetchTranscript(
          prompt
        );

        if (!transcriptResponse) {
          return res.status(400).json({ error: "Failed to get transcript" });
        }
        let transcript = "";

        transcriptResponse.forEach((line) => {
          transcript += line.text;
        });

        const response = await initializeChain(initialPrompt, transcript);

        return res.status(200).json({ output: response, chatHistory });
      } catch (err) {
        console.error(err);
        return res
          .status(500)
          .json({ error: "An error occurred while fetching transcript" });
      }
    } else {
      try {
        chatHistory.push({
          role: "user",
          content: prompt,
        });

        const response = await chain.call({
          question: prompt,
          chat_history: chatHistory,
        });

        chatHistory.push({
          role: "assistant",
          content: response.text,
        });

        return res.status(200).json({ output: response, chatHistory });
      } catch (error) {
        // Generic error handling
        console.error(error);
        res
          .status(500)
          .json({ error: "An error occurred during the conversation." });
      }
    }
  }
}
