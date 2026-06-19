from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from services import YouTubeLLMService

class SummarizeRequest(BaseModel):
    video_url: str

app = FastAPI(
    title="VideoIntelligenceAPI",
    description="Asynchronous processing engine to fetch transcripts and generate generative summaries.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

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

        yt_service = YouTubeLLMService(api_key=OPENAI_KEY)
        
        video_id = yt_service.extract_video_id(url)
        transcript = yt_service.get_formatted_transcript(video_id)
        summary_output = yt_service.generate_summary(transcript)
        
        return {
            "status": "success",
            "message": "Summary generated successfully.",
            "summary": summary_output
        }
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount frontend directory structures
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
