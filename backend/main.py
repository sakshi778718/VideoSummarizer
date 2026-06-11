from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# Import your custom service class from services.py
from services import YouTubeLLMService

# Define the expected structure from your frontend
class SummarizeRequest(BaseModel):
    video_url: str

app = FastAPI(
    title="VideoIntelligenceAPI",
    description="Asynchronous processing engine to fetch transcripts and generate generative summaries.",
    version="1.0.0"
)

# 1. ALLOW CORS (Cross-Origin Resource Sharing)
# Keeps your deployment safe from cross-origin browser blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fetch the OpenAI API key from Render's environment variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# 2. INTEGRATED BACKEND ENDPOINT
@app.post("/api/v1/summarize")
async def summarize_video(payload: SummarizeRequest):
    try:
        url = payload.video_url
        if not url:
            raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL.")
        
        if not OPENAI_KEY:
            raise HTTPException(
                status_code=500, 
                detail="OpenAI API Key is missing on the server configuration. Please check Render Environment Variables."
            )

        # Initialize your YouTubeLLMService with the active key
        yt_service = YouTubeLLMService(api_key=OPENAI_KEY)
        
        # Step 1: Extract Video ID using your custom staticmethod logic
        video_id = yt_service.extract_video_id(url)
        
        # Step 2: Fetch the transcript and pass it to your LLM summarizing function
        # (Assuming your method name below matches what you built further down in services.py)
        transcript = yt_service.get_formatted_transcript(video_id)
        
        # Step 3: Generate the summary using your service instance
        # Replace 'generate_summary_from_transcript' with your actual method name if it's different!
        summary_output = yt_service.generate_summary_from_transcript(transcript)
        
        return {
            "status": "success",
            "message": "Summary generated successfully.",
            "summary": summary_output
        }
        
    except HTTPException as http_err:
        # Pass through the HTTPExceptions handled directly inside your services.py methods
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. MOUNT FRONTEND DIRECTORY AND SERVE INDEX.HTML AT THE ROOT (/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "error": "Backend running, but frontend/index.html was not found.",
        "resolved_path": index_path
    }
