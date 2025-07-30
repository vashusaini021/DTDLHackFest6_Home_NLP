# 📊 Excel AI Query API

This project allows you to perform natural language queries on Excel files (even across multiple sheets and files) using OpenAI’s GPT models and LangChain with FAISS for semantic search.

It exposes a FastAPI backend that you can run locally and query with plain English to retrieve structured data.

---

## 🚀 Features

- Upload Excel files with multiple sheets
- Semantic/NLP querying via OpenAI (GPT-4o)
- Uses FAISS vector store for document indexing
- Smart AI-based field mapping
- Embedding & mapping cache to reduce API calls
- Throttled embeddings to respect OpenAI rate limits

---

## 📁 Project Structure


├── app.py                # Main FastAPI app

├── requirements.txt      # All dependencies

├── vectorstore/          # FAISS vector DB (generated automatically)

└── README.md             # You're reading it!


---
## 📦 Requirements

- Python 3.9+
- OpenAI API Key

Install dependencies:

```bash
python3 -m pip install --break-system-packages -r requirements.txt
---
