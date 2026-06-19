# VideoSummarizer

AI-powered YouTube Video Summarizer built with FastAPI, OpenAI, and YouTube Transcript API.

VideoSummarizer extracts transcripts from YouTube videos, organizes them into timestamped sections, and generates concise, structured summaries using GPT-4o Mini.

## Live Demo

https://videosummarizer-rdw7.onrender.com

---

## Features

- Extract YouTube Video IDs from multiple URL formats
- Retrieve video transcripts automatically
- Organize transcripts into timestamped chunks
- Generate AI-powered summaries
- Produce professional markdown reports
- FastAPI backend architecture
- Clean and responsive frontend interface

---

## Summary Output Format

The AI generates summaries in the following structure:


### Key Takeaways

- Important insights
- Major lessons
- Critical facts

### Chronological Deep Dive

Timestamp-based breakdown of the video content.

Example:

```markdown
# Executive Summary

This video explains modern AI trends and practical applications.

# Key Takeaways

- AI adoption is accelerating.
- Businesses benefit from automation.
- Responsible AI remains important.

# Chronological Deep Dive

## [00:00]
Introduction to AI.

## [02:00]
Discussion on machine learning.

## [04:00]
Real-world use cases.
```

---

## Tech Stack

### Backend

- Python
- FastAPI
- OpenAI API
- YouTube Transcript API

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Render

---

## Project Structure

```text
VideoSummarizer/
│
├── backend/
│   ├── main.py
│   ├── services/
│   │   └── youtube_llm_service.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/VideoSummarizer.git
cd VideoSummarizer
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the backend directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Run Backend

```bash
uvicorn main:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

---

## How It Works

1. User submits a YouTube URL.
2. Application extracts the YouTube video ID.
3. Transcript is fetched using YouTube Transcript API.
4. Transcript is grouped into timestamped chunks.
5. GPT-4o Mini analyzes the transcript.
6. A structured summary is generated and returned.

---

## Example Request

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

---

## Dependencies

```text
fastapi
uvicorn
openai
youtube-transcript-api
python-dotenv
```

Install manually:

```bash
pip install fastapi uvicorn openai youtube-transcript-api python-dotenv
```

---

## Error Handling

The application handles:

- Invalid YouTube URLs
- Missing OpenAI API keys
- Transcript retrieval failures
- Videos without captions
- OpenAI API errors
- Internal processing exceptions

---

## Future Enhancements

- Multi-language transcript support
- Download summaries as PDF
- Video chapter generation
- Mind-map generation
- Key quote extraction
- Speaker identification
- Adjustable summary length
- Batch video summarization

---

## Author

**Sakshi**

Built with FastAPI, OpenAI, and YouTube Transcript API.

---

## License

This project is licensed under the MIT License.
