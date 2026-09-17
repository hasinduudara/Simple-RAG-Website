import io
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# HuggingFace Models configuration
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_ID}/pipeline/feature-extraction"
LLM_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# Initialize FastAPI App
app = FastAPI(title="Dynamic PDF RAG API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "pdf_documents"


# Pydantic Model for Question Request Body
class QueryRequest(BaseModel):
    question: str


def require_settings():
    missing = [
        name
        for name, value in {
            "HF_API_KEY": HF_API_KEY,
            "QDRANT_URL": QDRANT_URL,
            "QDRANT_API_KEY": QDRANT_API_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing server configuration: {', '.join(missing)}",
        )


def get_qdrant_client():
    require_settings()
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# Helper Function: Extract text directly from uploaded PDF bytes
def extract_text_from_pdf_bytes(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


# Helper Function: Split text into smaller chunks
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# Helper Function: Get embedding vector from HuggingFace
def get_embedding(text):
    require_settings()
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    try:
        response = requests.post(
            EMBEDDING_API_URL, headers=headers, json=payload, timeout=20
        )
        response.raise_for_status()
        res_json = response.json()
        if (
            isinstance(res_json, list)
            and len(res_json) > 0
            and isinstance(res_json[0], list)
        ):
            return res_json[0]
        return res_json
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None


# Helper Function: Generate answer using LLM
def generate_answer(query, context):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user question based ONLY on the provided context. Be concise.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": messages,
        "max_tokens": 150,
    }

    try:
        response = requests.post(
            LLM_CHAT_URL, headers=headers, json=payload, timeout=30
        )
        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            return res_json["choices"][0]["message"]["content"]
        else:
            return "Sorry, I couldn't generate an answer."
    except Exception as e:
        print(f"LLM Error: {e}")
        return "An error occurred while generating the answer."


# --- API Endpoints ---


@app.get("/")
def read_root():
    return {"message": "RAG API is running!"}


# Endpoint 1: Upload PDF, process text, embed chunks, and update Qdrant
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Please upload a valid PDF file."
        )

    try:
        qdrant = get_qdrant_client()

        # Read file bytes
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf_bytes(file_bytes)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from this PDF.",
            )

        # Chunk extracted text
        chunks = chunk_text(extracted_text, chunk_size=300, overlap=50)

        # Re-create Qdrant collection to replace old document vectors
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Generate embeddings and upload points
        points = []
        for idx, chunk in enumerate(chunks, 1):
            vector = get_embedding(chunk)
            if vector:
                points.append(
                    PointStruct(id=idx, vector=vector, payload={"text": chunk})
                )

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "message": f"PDF '{file.filename}' processed and indexed successfully!",
            "total_chunks": len(points),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}"
        )


# Endpoint 2: Ask question based on current indexed PDF
@app.post("/ask")
def ask_question(request: QueryRequest):
    user_query = request.question
    qdrant = get_qdrant_client()

    # 1. Convert question to vector
    query_vector = get_embedding(user_query)
    if not query_vector:
        raise HTTPException(
            status_code=500, detail="Failed to generate query embedding."
        )

    # 2. Search Qdrant
    if not qdrant.collection_exists(COLLECTION_NAME):
        return {
            "answer": "No document indexed yet. Please upload a PDF first!"
        }

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=2
    ).points

    if not search_results:
        return {"answer": "No relevant information found in the document."}

    # 3. Create context and generate answer
    retrieved_texts = [hit.payload["text"] for hit in search_results]
    context = "\n---\n".join(retrieved_texts)

    final_answer = generate_answer(user_query, context)
    return {"answer": final_answer.strip()}
