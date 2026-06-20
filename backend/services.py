import os
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
        """Fetches video transcripts using adaptive instance fallback strategies to support new version specs."""
        transcript_list = None
        last_error = None
        
        # Determine cookies path if available
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cookies_path = os.path.join(current_dir, "youtube_cookies.txt")
        cookies = cookies_path if os.path.exists(cookies_path) else None

        # Instantiate the API client according to recent package specs
        yt_api = YouTubeTranscriptApi()

        # Strategy 1: Direct modern fetch instance call
        try:
            transcript_list = yt_api.fetch(video_id, languages=['en', 'hi'])
        except Exception as e:
            last_error = e

        # Strategy 2: Instance List enumeration fallback
        if not transcript_list:
            try:
                retrieved_list = yt_api.list(video_id)
                try:
                    transcript_obj = retrieved_list.find_transcript(['en', 'es', 'hi'])
                except Exception:
                    transcript_obj = next(iter(retrieved_list))
                    
                transcript_list = transcript_obj.fetch()
            except Exception as e:
                last_error = e

        # Final Exception Guard
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
            text_content = entry.text if hasattr(entry, 'text') else entry.get('text', '')
            start_val = entry.start if hasattr(entry, 'start') else entry.get('start', 0.0)

            if not chunk_text:
                start_time = start_val
            
            chunk_text.append(text_content)
            
            # Chunk transcript segments every 120 seconds to maintain contextual relevance
            if start_val - start_time > 120:
                timestamp = self.format_time(start_time)
                formatted_segments.append(f"[{timestamp}] {' '.join(chunk_text)}")
                chunk_text = []
        
        if chunk_text:
            formatted_segments.append(f"[{self.format_time(start_time)}] {' '.join(chunk_text)}")
            
        return "\n\n".join(formatted_segments)

    def generate_summary(self, transcript: str, dimension: str = "executive") -> str:
        """Invokes the LLM to synthesize raw text into specialized visual and analytical layout formats."""
        try:
            # Multi-dimension prompting maps based on choice selection
            dimensions_prompt = {
                "executive": (
                    "- **Executive Summary (TL;DR)**: A punchy 2-3 sentence high-level overview.\n"
                    "- **Key Takeaways**: Bulleted items highlighting core data and lessons.\n"
                    "- **Chronological Deep Dive**: Breakdown the narrative flow referencing the [MM:SS] timestamps."
                ),
                "educational": (
                    "- **Core Definitions**: Technical terms and concepts mapped out clearly.\n"
                    "- **Step-by-Step Concepts**: Deconstructed complex concepts explained linearly.\n"
                    "- **Active Recall Q&A**: Five key questions and flashcard-style answers derived from the content."
                ),
                "actionable": (
                    "- **Direct Directives**: What concrete actions are specified by the speaker?\n"
                    "- **Task Matrix Checklist**: Clean checklist format explicitly referencing exact [MM:SS] points.\n"
                    "- **Core Resource Index**: Mentions of tools, books, frameworks or external citations."
                )
            }

            system_instruction = (
                "You are an elite research analyst. Synthesize the provided timestamped YouTube transcript "
                "into a high-density, professional summary report. Format your output strictly using Markdown.\n\n"
                f"Structure the report layout precisely around this dimension option group:\n{dimensions_prompt.get(dimension, dimensions_prompt['executive'])}\n\n"
                "CRITICAL EXTRA INSTRUCTION: At the absolute bottom of your entire response, add a section header exactly named "
                "`## Visual Mind Map Representation`. Below that header, write a valid, beautiful nested Markdown list structure "
                "representing the core map tree topology (using single hyphens, indenting child items with tabs/spaces) summarizing "
                "the entire subject matter hierarchy. This tree will be parsed directly into an interactive graphic diagram node chart."
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
