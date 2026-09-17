# 🚀 Smart Dynamic RAG Assistant

A full-stack Retrieval-Augmented Generation (RAG) web application built with **FastAPI**, **Qdrant Vector Database**, and **Hugging Face (Llama 3.1 & BGE Embeddings)**. This application allows users to dynamically upload PDF documents and chat with an AI assistant that answers questions based exclusively on the uploaded content.

---

👉 Live - https://simple-rag-website.vercel.app/

---

## 🛠️ Tech Stack

* **Backend:** FastAPI (Python), Uvicorn, Pydantic, Pypdf
* **Vector Database:** Qdrant Cloud
* **AI / ML Models (via Hugging Face Inference API):**
  * **Embeddings:** `BAAI/bge-small-en-v1.5` (384 dimensions)
  * **LLM:** `meta-llama/Llama-3.1-8B-Instruct`
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla), Fetch API

---

## ⚙️ Architecture & Data Pipeline

1. **Document Ingestion & Chunking:** Uploaded PDFs are parsed directly in memory (`RAM`) using `pypdf`. The extracted text is split into manageable chunks with an overlap to preserve contextual continuity. The browser uploads PDFs through the JSON `/api/upload-json` endpoint and limits raw PDF files to 3 MB so the base64 request body stays under Vercel's request size limit.
2. **Embedding Generation:** Each text chunk is sent to the Hugging Face Inference API to generate dense vector representations.
3. **Vector Storage & Similarity Search:** Vectors and text payloads are indexed in **Qdrant Cloud**. When a query is made, Qdrant performs a cosine similarity search to retrieve the most relevant text chunks.
4. **Context-Aware Generation:** The retrieved chunks are injected into a prompt template and sent to **Meta Llama 3.1**, which formulates a precise, grounded answer.

---

## 📁 Project Structure

```text
rag-pdf-assistant/
│
├── backend/
│   ├── main.py          # FastAPI application & RAG pipeline logic
│   ├── requirements.txt # Python dependencies
│   └── .env             # API keys and environment variables (ignored in Git)
│
├── frontend/
│   └── index.html       # Web UI for file upload and interactive chat
│
└── README.md            # Project documentation
