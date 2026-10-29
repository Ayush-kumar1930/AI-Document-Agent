import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from text_processor import extract_text_from_pdf
from vector_store import store_chunks, reset_collection
from rag_engine import generate_answer

app = FastAPI(title="AI Document Agent (RAG)", version="1.0")

# Request schema for asking questions
class QuestionRequest(BaseModel):
    question: str


# Endpoint 1: Upload and Process PDF with Page Breadcrumbs
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Validates that the uploaded file is a PDF.
    2. Saves it temporarily.
    3. Extracts chunks page-by-page (preserving page numbers for breadcrumbs).
    4. Clears previous vector DB collection and stores new chunks with page metadata.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily to disk
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = os.path.join("./temp_uploads", file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract chunks with page metadata directly
        # Returns: [{"text": "...", "page": 1}, ...]
        chunks_data = extract_text_from_pdf(temp_path, chunk_size=500, chunk_overlap=100)
        
        if not chunks_data:
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no readable text.")

        # Reset old vector DB data and store new document chunks with page breadcrumbs
        reset_collection()
        store_chunks(chunks_data)

        total_chars = sum(len(c["text"]) for c in chunks_data)

        return {
            "status": "success",
            "filename": file.filename,
            "total_characters": total_chars,
            "total_chunks": len(chunks_data),
            "message": f"Successfully processed '{file.filename}' into {len(chunks_data)} chunks with page breadcrumbs!"
        }
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Endpoint 2: Ask a Question (Routed by Intent Classifier)
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Accepts a question, routes through the Intent Classifier,
    and returns the grounded answer along with source page citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    result = generate_answer(request.question)
    return result


# Endpoint 3: Serve the Web Interface
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Returns the single-page HTML/CSS/JS frontend."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Endpoint 4: Clear ChromaDB.
@app.post("/reset")
async def reset_db():
    """Wipes the database clean."""
    reset_collection()
    return {"status": "cleared", "message": "All documents removed from memory."}
