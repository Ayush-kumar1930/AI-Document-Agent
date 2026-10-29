# AI Document Agent (Multilingual RAG System)

A production-grade, full-stack Document Q&A Agent built with **FastAPI**, **ChromaDB**, **Sentence-Transformers**, and **OpenRouter (Llama-3.3-70B / Llama-3.1-8b)**.

## Features
- **Strict Document Grounding:** Never hallucinates; answers strictly using facts from the uploaded PDF.
- **Multilingual & Hinglish Preprocessor:** Understands queries in English, Hinglish, Hindi, and regional languages. Automatically reformulates queries for precise vector retrieval.
- **Page Breadcrumbs & Citations:** Cites exact page numbers (e.g., `📍 Page 2`) for every extracted fact.
- **Smart Guided Fallbacks:** Suggests 3 clickable related questions whenever an out-of-scope question is asked.
- **Natural Conversational Persona:** Handles greetings, pleasantries ("thank you", "okh"), and casual chat dynamically.
- **Interactive Web Interface:** Single-page dashboard with drag-and-drop PDF upload and live Markdown/table rendering.

---

## Architecture

```
[ User Query (English / Hinglish / Hindi) ]
                     │
                     ▼
       [ LLM Query Preprocessor (8B) ]
       ├── Classifies Intent (CONVERSATIONAL, SUMMARY, FACTUAL)
       ├── Detects Language
       └── Translates into English Search Keywords
                     │
                     ▼
       [ ChromaDB Persistent Vector Store ]
       ──▶ Sub-millisecond Cosine Similarity Search
                     │
                     ▼
       [ Grounded Generation (70B) ]
       ──▶ Accurate Answer + Page Breadcrumbs + Dynamic Suggestions
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and paste your OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Run the Application
```bash
uvicorn main:app --reload
```
Open your browser at **`http://localhost:8000`**.

---

## License
MIT License

