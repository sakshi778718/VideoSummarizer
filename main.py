import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from services import YouTubeLLMService

load_dotenv()

app = FastAPI(
    title="VideoIntelligenceAPI",
    description="Asynchronous processing engine to fetch transcripts and generate generative summaries.",
    version="1.0.0"
)

# Essential CORS configurations to allow the local front-end client to securely invoke endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    url: str

# Dependency Injection pattern for modular, decoupled testing
def get_llm_service():
    return YouTubeLLMService(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/api/v1/summarize", status_code=200)
async def summarize_endpoint(payload: SummarizeRequest, service: YouTubeLLMService = Depends(get_llm_service)):
    """
    Ingests a YouTube URL, processes the transcript stream asynchronously, 
    and pipes data through OpenAI to build structured, timestamped markdown summaries.
    """
    video_id = service.extract_video_id(payload.url)
    transcript_data = service.get_formatted_transcript(video_id)
    summary_markdown = service.generate_summary(transcript_data)
    
    return {
        "status": "success",
        "video_id": video_id,
        "data": {
            "summary": summary_markdown
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
