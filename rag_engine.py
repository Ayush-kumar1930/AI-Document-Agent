import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import search_similar_chunks, get_first_chunks, get_document_count

# 1. Load the OpenRouter API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing! Please set it in your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "FastAPI Multilingual RAG Agent",
    }
)

FAST_MODEL = "meta-llama/llama-3.1-8b-instruct"
MAIN_MODEL = "meta-llama/llama-3.3-70b-instruct"


def preprocess_query(user_question: str) -> dict:
    """
    Universal Multilingual & Hinglish Preprocessor:
    1. Classifies intent (CONVERSATIONAL, SUMMARY, FACTUAL) regardless of language/slang.
    2. Identifies the user's language (English, Hinglish, Hindi, etc.).
    3. Translates code-mixed queries into clean English keywords for ChromaDB!
    """
    system_prompt = (
        "You are an expert conversational AI query understanding agent for a document assistant.\n"
        "Analyze the user's input and return a STRICT JSON object with these keys:\n"
        "1. 'intent':\n"
        "   - 'CONVERSATIONAL': for greetings (hi, hello), pleasantries (thanks, thank you, okh, okay, cool, great, shukriya), "
        "or identity/casual chat (what are you doing, who are you, kaise ho, what's up).\n"
        "   - 'SUMMARY': for explain/summarize document, overview, main points, kya hai isme.\n"
        "   - 'FACTUAL': for specific questions about facts, terms, numbers, policies.\n"
        "2. 'language': Set 'English' if the input is in English. Set 'Hindi' or 'Hinglish' ONLY IF the user explicitly used Hindi words (kya, kaise, batao, shukriya, namaste, etc.). NEVER guess Hindi if the input is in English!\n"
        "3. 'english_search_query': Concise English search keywords for searching technical documents.\n\n"
        "Return ONLY raw JSON, with no markdown code fences or extra text."
    )
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.0,
            max_tokens=80,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*|^```\s*|```$", "", content).strip()
        return json.loads(content)
    except Exception:
        return {
            "intent": "FACTUAL",
            "language": "English",
            "english_search_query": user_question
        }


def parse_suggestions(raw_text: str) -> tuple[str, list[str]]:
    """Extracts suggestions if the model included 'SUGGESTIONS:' at the end of its response."""
    if "SUGGESTIONS:" in raw_text:
        parts = raw_text.split("SUGGESTIONS:")
        clean_answer = parts[0].strip()
        suggestions_block = parts[1].strip()

        suggestions = []
        for line in suggestions_block.split("\n"):
            line = re.sub(r"^[-*0-9.]+\s*", "", line).strip()
            if line:
                suggestions.append(line)
        return clean_answer, suggestions[:3]
    return raw_text.strip(), []


def generate_answer(question: str) -> dict:
    doc_count = get_document_count()

    # Step A: Multilingual Query Understanding & Normalization
    query_info = preprocess_query(question)
    intent = query_info.get("intent", "FACTUAL")
    lang = query_info.get("language", "English")
    english_query = query_info.get("english_search_query", question)

    # --- 1. CONVERSATIONAL & PLEASANTRIES (Dynamic, Warm & Human!) ---
    if intent in ["CONVERSATIONAL", "GREETING"]:
        conv_system_prompt = (
            f"You are a warm, witty, and exceptionally helpful AI Document Assistant. "
            f"Respond naturally to the user in their exact language ({lang}).\n"
            "RULES:\n"
            "1. If they say 'thank you' or 'thanks': warmly reply that you're happy to help!\n"
            "2. If they say 'okh', 'cool', 'got it': acknowledge cheerfully.\n"
            "3. If they ask 'what are you doing' or 'who are you': explain your role warmly in 1-2 sentences.\n"
            "4. Keep it brief, natural, and friendly (1 to 2 sentences max). Do NOT lecture.\n"
            "5. At the end, include 3 helpful question suggestions formatted as:\n"
            "SUGGESTIONS:\n"
            "- <Suggested question 1>\n"
            "- <Suggested question 2>\n"
            "- <Suggested question 3>"
        )
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": conv_system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
        )
        raw_answer = response.choices[0].message.content
        clean_answer, suggestions = parse_suggestions(raw_answer)

        return {
            "answer": clean_answer,
            "sources": [],
            "suggestions": suggestions
        }

    # --- 2. CHECK IF DOCUMENT EXISTS FOR QUESTIONS ---
    if doc_count == 0:
        msg = (
            "Aapne abhi tak koi document upload nahi kiya hai. Kripya pehle PDF upload karein!"
            if "hindi" in lang.lower() or "hinglish" in lang.lower()
            else "No document has been uploaded yet. Please upload a PDF on the left panel first!"
        )
        return {
            "answer": msg,
            "sources": [],
            "suggestions": []
        }

    # --- 3. GLOBAL SUMMARY ---
    if intent == "SUMMARY":
        first_chunks = get_first_chunks(n=4)
        search_chunks = search_similar_chunks("overview conclusion summary", n_results=2)
        combined = {c["text"]: c for c in (first_chunks + search_chunks)}
        sources = list(combined.values())

        system_prompt = (
            f"You are a helpful document assistant. The user wants an overview/summary of the document.\n"
            f"Respond in the user's language: {lang}.\n"
            "Based only on the provided Context (which includes the beginning pages and key sections), "
            "provide a structured, executive summary explaining what this document is about, its purpose, "
            "and core topics.\n"
            "State clearly at the beginning what the document is."
        )
        temp = 0.2

    # --- 4. FACTUAL QUESTION WITH SMART FALLBACKS ---
    else:
        sources = search_similar_chunks(english_query, n_results=4)

        system_prompt = (
            f"You are a strict, helpful document assistant. Answer using ONLY the facts "
            f"in the Context below. Respond in the user's language: {lang}.\n\n"
            "Rules for High User Satisfaction:\n"
            "1. If the answer is in the Context, answer clearly, concisely, and cite page numbers (e.g. 'According to Page 2...').\n"
            "2. If the user's question is UNRELATED or NOT found in the document (like weather, pasta, personal life):\n"
            f"   - Politely explain in {lang} that this document does not cover that topic.\n"
            f"   - Offer a warm alternative: 'However, here are some topics covered in this document you might find interesting:'\n"
            "   - End your response with 3 relevant questions based on what IS in the Context, formatted EXACTLY like this:\n"
            "SUGGESTIONS:\n"
            "- <Question 1 about the document>\n"
            "- <Question 2 about the document>\n"
            "- <Question 3 about the document>\n"
            "3. Do NOT use outside knowledge."
        )
        temp = 0.0

    if not sources:
        return {
            "answer": "No documents found. Please upload a PDF first.",
            "sources": [],
            "suggestions": []
        }

    # Format context with Page Breadcrumbs!
    context_text = "\n\n---\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in sources])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temp,
    )

    raw_answer = response.choices[0].message.content
    clean_answer, suggestions = parse_suggestions(raw_answer)

    # Broad refusal detection to ensure NO misleading sources are ever shown!
    refusal_keywords = [
        "cannot find", "could not find", "not in the uploaded", "not in the provided",
        "outside the scope", "not related", "can't answer", "cannot answer",
        "i'm afraid", "does not contain", "does not discuss", "not mentioned in",
        "does not cover", "unable to find", "no information",
        "maaf kijiye", "maaf kijye", "yeh vishay", "dastavez mein nahi",
        "माफ़ कीजिये", "माफ़ करें", "दस्तावेज़ में नहीं", "उपलब्ध नहीं"
    ]

    ans_lower = clean_answer.lower()
    is_refusal = any(k in ans_lower for k in refusal_keywords)

    return {
        "answer": clean_answer,
        "sources": [] if is_refusal else sources,
        "suggestions": suggestions
    }
