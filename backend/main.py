from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models.user import User
import logging
import requests
from pydantic import BaseModel
import json
import re
from .utils.resume_parser import parse_and_chunk_resume
from .database.init_db import init_db, get_db
import pinecone
from contextlib import asynccontextmanager
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("asha_chatbot.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

if not asyncio.iscoroutinefunction(init_db):
    logger.error("init_db is not a coroutine function")
    raise ImportError("init_db from database.init_db is not a valid async function")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Attempting to initialize database")
    await init_db()
    logger.info("Database initialized on startup")
    yield
    logger.info("Shutdown event triggered")

app = FastAPI(title="ASHA AI Chatbot", lifespan=lifespan)

# Fix CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (tighten later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobQuery(BaseModel):
    query: str

class GuidanceRequest(BaseModel):
    user_id: str

async def get_db_connection():
    async with get_db() as conn:
        yield conn

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("Root route accessed")
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    logger.info("health check passed")
    return {"status": "healthy"}

@app.post("/job-search")
async def job_search(job_query: JobQuery, db=Depends(get_db_connection)):
    try:
        query = job_query.query.lower().strip()
        cursor = await db.execute("SELECT skills, preferences FROM users WHERE user_id = 'temp_user'")
        user_data = await cursor.fetchone()
        user_data = {"skills": user_data[0], "preferences": user_data[1]} if user_data else {"skills": None, "preferences": None}
        parsed_query = {"role": "any", "location": "global", "experience": "any"}

        if query:
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}"
            gemini_response = requests.post(
                gemini_url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": f"Extract role, location, experience from: '{query}'. Infer role based on skills mentioned (e.g., 'ml' for machine learning, 'genai' for generative ai). Return as JSON e.g. {{'role': 'internship in machine learning and generative ai', 'location': 'unspecified', 'experience': 'entry level'}}"}]}]}
            )
            gemini_response.raise_for_status()
            data = gemini_response.json()
            raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            logger.info(f"Raw Gemini response: {raw_text}")
            clean_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE).strip()
            try:
                parsed_query.update(json.loads(clean_text) if clean_text else {})
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from Gemini: {clean_text}")
                parsed_query = {"role": "internship", "location": "global", "experience": "entry level"}
            logger.info(f"LLM parsed query: {parsed_query}")
        else:
            parsed_query = {"role": "any job", "location": "global", "experience": "any"}

        if user_data["skills"] and user_data["preferences"]:
            parsed_query["role"] = f"{user_data['skills']} {parsed_query['role']}"
            parsed_query["location"] = user_data["preferences"] or parsed_query["location"]

        jsearch_url = "https://jsearch.p.rapidapi.com/search"
        headers = {"X-RapidAPI-Key": settings.JSEARCH_API_KEY, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
        params = {"query": f"{parsed_query.get('role', 'jobs')} in {parsed_query.get('location', 'global')}", "num_pages": 1, "date_posted": "all"}
        logger.info(f"JSearch request params: {params}")
        jsearch_response = requests.get(jsearch_url, headers=headers, params=params)
        jsearch_response.raise_for_status()
        results = jsearch_response.json().get("data", [])
        logger.info(f"JSearch returned {len(results)} results for query: {query}")
        return {"results": results[:10]}
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        return {"error": "API failed, check logs"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {"error": "Failed to parse LLM response"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Something went wrong, try again"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), db=Depends(get_db_connection)):
    try:
        content = await file.read()
        chunks = parse_and_chunk_resume(content, settings.COHERE_API_KEY, settings.PINECONE_API_KEY)
        logger.info(f"Processed resume into {len(chunks)} chunks")
        await db.execute("INSERT OR REPLACE INTO users (user_id, skills, preferences) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET skills=excluded.skills, preferences=excluded.preferences", ("temp_user", "machine learning, generative ai", "global"))
        await db.commit()
        return {"message": "Resume processed and chunked successfully", "chunk_count": len(chunks)}
    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        return {"error": "Failed to process resume, check logs"}

@app.post("/personalized-guidance")
async def personalized_guidance(request: GuidanceRequest):
    pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index("asha-resume-chunks")
    result = index.query(vector=[0] * 1024, top_k=3, include_metadata=True)
    if not result.matches:
        return {"message": "No resume data found"}
    resume_text = " ".join([match.metadata["text"].replace("\n", " ") for match in result.matches])
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}"
    response = requests.post(gemini_url, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": f"Provide career guidance based on this resume summary: {resume_text}. Suggest specific skills to improve and roles to target."}]}]})
    response.raise_for_status()
    guidance = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Enhance your skills based on your experience.")
    return {"guidance": guidance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
