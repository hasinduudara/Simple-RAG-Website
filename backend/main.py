import base64
import io
import os
import requests
import time
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
MAX_UPLOAD_CHUNKS = int(os.getenv("MAX_UPLOAD_CHUNKS", "20"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))

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


class UploadRequest(BaseModel):
    filename: str
    file_base64: str


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
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=30,
    )


def service_error(service, action, error):
    print(f"{service} {action} Error: {error}")
    raise HTTPException(
        status_code=502,
        detail=f"{service} {action} failed. Please try again. Details: {error}",
    )


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


def validate_pdf_upload(filename, file_bytes):
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Please upload a valid PDF file."
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"PDF is too large. Please upload a file smaller than {max_mb:.1f} MB.",
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="The selected file does not look like a valid PDF.",
        )


def process_pdf_bytes(filename, file_bytes):
    validate_pdf_upload(filename, file_bytes)
    print(f"Upload Step: received {filename} ({len(file_bytes)} bytes)")

    extracted_text = extract_text_from_pdf_bytes(file_bytes)
    print(f"Upload Step: extracted {len(extracted_text)} characters")

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text from this PDF.",
        )

    chunks = chunk_text(extracted_text)
    print(f"Upload Step: created {len(chunks)} chunks")

    qdrant = get_qdrant_client()

    try:
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    except Exception as e:
        service_error("Qdrant", "collection setup", e)

    points = []
    for idx, chunk in enumerate(chunks, 1):
        vector = get_embedding(chunk)
        if vector:
            points.append(
                PointStruct(id=idx, vector=vector, payload={"text": chunk})
            )

    if not points:
        raise HTTPException(
            status_code=502,
            detail="Could not generate embeddings for this PDF.",
        )

    try:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        service_error("Qdrant", "vector upload", e)

    return {
        "message": f"PDF '{filename}' processed and indexed successfully!",
        "total_chunks": len(points),
        "indexed_characters": len("".join(chunks)),
    }


# Helper Function: Split text into smaller chunks
def chunk_text(text, chunk_size=800, overlap=100, max_chunks=MAX_UPLOAD_CHUNKS):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length and len(chunks) < max_chunks:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# Helper Function: Get embedding vector from HuggingFace
def get_embedding(text):
    require_settings()
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                EMBEDDING_API_URL, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            res_json = response.json()
            if (
                isinstance(res_json, list)
                and len(res_json) > 0
                and isinstance(res_json[0], list)
            ):
                vector = res_json[0]
            else:
                vector = res_json

            if isinstance(vector, list) and len(vector) == 384:
                return vector

            raise ValueError(f"Unexpected embedding response shape: {res_json}")
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(1 + attempt)

    service_error("Hugging Face", "embedding request", last_error)


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
@app.get("/api")
def read_root():
    return {"message": "RAG API is running!"}


# Endpoint 1: Upload PDF, process text, embed chunks, and update Qdrant
@app.post("/upload")
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        return process_pdf_bytes(file.filename, file_bytes)
    except OSError as e:
        print(f"Upload stream Error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Upload stream failed before the PDF could be read: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/upload-json")
@app.post("/api/upload-json")
async def upload_pdf_json(request: UploadRequest):
    try:
        try:
            file_bytes = base64.b64decode(request.file_base64, validate=True)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid PDF upload data: {e}"
            )

        return process_pdf_bytes(request.filename, file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload JSON Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}"
        )


# Endpoint 2: Ask question based on current indexed PDF
@app.post("/ask")
@app.post("/api/ask")
def ask_question(request: QueryRequest):
    user_query = request.question
    qdrant = get_qdrant_client()

    # 1. Convert question to vector
    query_vector = get_embedding(user_query)
    # 2. Search Qdrant
    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            return {
                "answer": "No document indexed yet. Please upload a PDF first!"
            }

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=2
        ).points
    except Exception as e:
        service_error("Qdrant", "search", e)

    if not search_results:
        return {"answer": "No relevant information found in the document."}

    # 3. Create context and generate answer
    retrieved_texts = [hit.payload["text"] for hit in search_results]
    context = "\n---\n".join(retrieved_texts)

    final_answer = generate_answer(user_query, context)
    return {"answer": final_answer.strip()}
