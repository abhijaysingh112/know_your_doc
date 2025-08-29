📚 Know Your Doc

Know Your Doc is an AI-powered Streamlit app that allows you to upload any PDF document, process its contents into vector embeddings, and then ask natural language questions about the document. The app uses FAISS for similarity search, Sentence Transformers for embeddings, and Google Gemini models for contextual answering.

✨ Features

📄 Upload & Process PDF – Extracts text from PDFs and splits into chunks

🔎 Vector Search with FAISS – Finds the most relevant text chunks for a query

🤖 AI-Powered Answers – Uses Google Gemini API to generate contextual responses

🎨 Dark Mode UI – Sleek design with styled answer & context cards

📋 Suggested Q&A – Example prompts to help you get started

🛠️ Tech Stack

Streamlit
 – UI framework

PyPDF2
 – PDF text extraction

SentenceTransformers
 – Embedding model (all-MiniLM-L6-v2)

FAISS
 – Vector similarity search

Google Gemini
 – Answer generation (RAG)
