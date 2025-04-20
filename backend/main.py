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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

        # Convert natural language to JSearch query using Gemini
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}"
        gemini_prompt = f"""You are an expert at converting natural language requests into precise JSearch API queries. Your task is to analyze the given natural language input and generate a valid JSearch API query as a URL query string that accurately reflects the user's intent, based on the provided JSearch API query parameters. Follow these steps:

1. Identify the key components of the natural language request, such as job title, location, country, date posted, employment types, work-from-home preference, job requirements, radius, excluded publishers, and fields to include.
2. Map these components to the appropriate JSearch API query parameters (e.g., `query`, `country`, `date_posted`, `employment_types`, `work_from_home`, `job_requirements`, `radius`, `exclude_job_publishers`, `fields`, `page`, `num_pages`).
3. Ensure the query adheres to JSearch API conventions, including proper parameter names, value formats, and constraints (e.g., allowed values for `date_posted`, `country`, `employment_types`).
4. Handle ambiguities by making reasonable assumptions based on common job search patterns, and include a brief comment explaining any assumptions made.
5. Output the JSearch API query as a URL query string (e.g., `query=developer+jobs+in+chicago&country=us`).

### JSearch API Query Parameters
- **query** (required): Free-form jobs search query (e.g., "developer jobs in chicago"). Include job title and location when possible.
- **page** (optional): Page number to return (1-100, default: 1).
- **num_pages** (optional): Number of pages to return (1-20, default: 1).
- **country** (optional): ISO 3166-1 alpha-2 country code (e.g., "us", default: "us").
- **language** (optional): ISO 639 language code (e.g., "en"). Defaults to primary language of the country.
- **date_posted** (optional): Time frame for job postings ("all", "today", "3days", "week", "month", default: "all").
- **work_from_home** (optional): Set to `true` for remote jobs (default: `false`).
- **employment_types** (optional): Comma-separated list of employment types ("FULLTIME", "CONTRACTOR", "PARTTIME", "INTERN").
- **job_requirements** (optional): Comma-separated list of requirements ("under_3_years_experience", "more_than_3_years_experience", "no_experience", "no_degree").
- **radius** (optional): Distance from location in kilometers.
- **exclude_job_publishers** (optional): Comma-separated list of publishers to exclude (e.g., "BeeBe,Dice").
- **fields** (optional): Comma-separated list of job fields to include (e.g., "employer_name,job_title").

Now, convert the following natural language request into a JSearch API query: "{query}" """
        gemini_response = requests.post(
            gemini_url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": gemini_prompt.format(query=query)}]}]}
        )
        gemini_response.raise_for_status()
        data = gemini_response.json()
        raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        logger.info(f"Raw Gemini response for query conversion: {raw_text}")
        clean_text = re.sub(r'^```.*\n|\n.*```$', '', raw_text, flags=re.MULTILINE).strip()
        try:
            jsearch_query = clean_text.split("\n")[0] if clean_text else "query=internships"
        except IndexError:
            jsearch_query = "query=internships"

        # Fetch jobs with the converted query
        jsearch_url = "https://jsearch.p.rapidapi.com/search"
        headers = {"X-RapidAPI-Key": settings.JSEARCH_API_KEY, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
        params = {k: v for k, v in [param.split('=') for param in jsearch_query.split('&')] if v}
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
