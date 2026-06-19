import os
import re
from fastapi import HTTPException
import youtube_transcript_api
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
        """Fetches video transcripts using adaptive fallback strategies to handle environment and version updates."""
        transcript_list = None
        last_error = None
        
        # Determine cookies path if available
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cookies_path = os.path.join(current_dir, "youtube_cookies.txt")
        cookies = cookies_path if os.path.exists(cookies_path) else None

        # Strategy 1: Direct Class Method Call (Standard format)
        if not transcript_list:
            try:
                if cookies:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, cookies=cookies)
                else:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                last_error = e

        # Strategy 2: Direct Module Function Reference Wrapper
        if not transcript_list:
            try:
                if hasattr(youtube_transcript_api, 'get_transcript'):
                    if cookies:
                        transcript_list = youtube_transcript_api.get_transcript(video_id, cookies=cookies)
                    else:
                        transcript_list = youtube_transcript_api.get_transcript(video_id)
            except Exception as e:
                last_error = e

        # Strategy 3: Advanced Modern Listing Enumeration Fallback
        if not transcript_list:
            try:
                # Try finding through list_transcripts
                get_list_method = None
                if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                    get_list_method = YouTubeTranscriptApi.list_transcripts
                elif hasattr(youtube_transcript_api, 'list_transcripts'):
                    get_list_method = youtube_transcript_api.list_transcripts
                
                if get_list_method:
                    if cookies:
                        retrieved_list = get_list_method(video_id, cookies=cookies)
                    else:
                        retrieved_list = get_list_method(video_id)
                    
                    # Find any english or primary language translation stream
                    try:
                        transcript_obj = retrieved_list.find_transcript(['en', 'es', 'hi'])
                    except:
                        # Fallback default directly picking the first available item inside list
                        transcript_obj = next(iter(retrieved_list))
                        
                    transcript_list = transcript_obj.fetch()
            except Exception as e:
                last_error = e

        # Final Exception Guard: If all collection variants fail, inform the client container
        if not transcript_list:
            raise HTTPException(
                status_code=422, 
                detail=f"Transcript engine exhausted all structural methods. Captions might be fully disabled or blocked. Internal Error: {str(last_error)}"
            )

        # Process the successfully parsed text chunks chronologically
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
