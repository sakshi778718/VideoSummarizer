import re
from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

class YouTubeLLMService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API Key is required but missing.")
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Parses and extracts the 11-character YouTube video ID."""
        pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?)/|\S*?[?&]v=)|youtu\.be/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")

    @staticmethod
    def format_time(seconds: float) -> str:
        """Formats raw floating seconds into neat MM:SS timestamps."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def get_formatted_transcript(self, video_id: str) -> str:
        """Fetches and builds a clean, timestamp-chunked layout of the video transcript using contemporary library methods."""
        try:
            # Instantiate the class and fetch the transcript to handle modern versions smoothly
            api_instance = YouTubeTranscriptApi()
            transcript_list = api_instance.fetch(video_id)
            
            formatted_segments = []
            chunk_text = []
            start_time = 0.0
            
            for entry in transcript_list:
                if not chunk_text:
                    start_time = entry['start']
                
                chunk_text.append(entry['text'])
                
                # Chunk transcript segments every 120 seconds to maintain contextual relevance
                if entry['start'] - start_time > 120:
                    timestamp = self.format_time(start_time)
                    formatted_segments.append(f"[{timestamp}] {' '.join(chunk_text)}")
                    chunk_text = []
            
            if chunk_text:
                formatted_segments.append(f"[{self.format_time(start_time)}] {' '.join(chunk_text)}")
                
            return "\n\n".join(formatted_segments)
            
        except Exception as e:
            raise HTTPException(
                status_code=422, 
                detail=f"Could not retrieve transcript. Captions may be disabled on this video. Error: {str(e)}"
            )

    def generate_summary(self, transcript: str) -> str:
        """Invokes the LLM to synthesize raw structured text into an executive summary."""
        try:
            system_instruction = (
                "You are an elite research analyst. Synthesize the provided timestamped YouTube transcript "
                "into a high-density, professional summary report. Format your output strictly using Markdown:\n"
                "- **Executive Summary (TL;DR)**: A punchy 2-3 sentence high-level overview.\n"
                "- **Key Takeaways**: Bulleted items highlighting core data and lessons.\n"
                "- **Chronological Deep Dive**: Breakdown the narrative flow referencing the [MM:SS] timestamps."
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Transcript Data:\n{transcript}"}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI synthesis engine execution failure: {str(e)}")
