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

```
├── app.py                # Main FastAPI app
├── requirements.txt      # All dependencies
├── vectorstore/          # FAISS vector DB (generated automatically)
└── README.md             # You're reading it!
```

---

## 📦 Requirements

- Python 3.9+
- OpenAI API Key

Install dependencies:

```bash
python3 -m pip install --break-system-packages -r requirements.txt

after successfully done with above install certify using below command

python3 -m pip install certifi --break-system-packages

```

---

## 🗃 Git LFS Setup (for large files)

GitHub restricts files >100MB from being pushed through regular Git. To handle large files (like FAISS index files), you must configure Git LFS (Large File Storage).

### 🔧 Install Git LFS

#### macOS:

```bash
brew install git-lfs
afte this clone repo with "git-lfs clone url.."
```

#### Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git-lfs
```

#### Windows:

- Download and install from: https://git-lfs.com/

---

## 🔑 OpenAI API Key

Set your OpenAI API key in `app.py` (look for `API_KEY = ...`), or modify the code to load from environment variables.

---

## ▶️ Run the App

Start the FastAPI server locally:

```bash
uvicorn app:app --reload
```

Then go to:
👉 **http://127.0.0.1:8000/docs** – to try it out via Swagger UI

---

## 🧠 How It Works

- Excel sheets are loaded and indexed
- LangChain splits the content and creates embeddings
- FAISS stores the vector data locally
- Queries are semantically matched using GPT and filtered
- Responses are returned as structured JSON

---

## 🧩 TODO / Ideas

- ✅ Smart semantic query understanding
- ✅ Vector caching using FAISS
- ⏳ File upload endpoint
- ⏳ Basic frontend (streamlit or React)
- ⏳ Long-term persistent vector DB

---

## 📝 License

This project is intended for internal/demo use. No warranties.
