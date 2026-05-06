# EduTutor — AI Learning Companion

> An open-source, multilingual AI tutoring system powered by Groq + Gemini Vision + FastAPI + a custom HTML/CSS/JS frontend.

![Stack](https://img.shields.io/badge/Stack-Groq%20%2B%20Gemini%20%2B%20FastAPI-e8b84b?style=flat-square)
![Cost](https://img.shields.io/badge/Cost-Free%20Tier%20Friendly-4ade80?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-8b7cf8?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-7%20Supported-60a5fa?style=flat-square)

---

## What is EduTutor?

EduTutor is a personalized AI tutoring system designed for students who want clear, example-driven explanations of any topic — supporting 7 languages, voice-based lessons, image/document understanding, and a beautiful dark academic UI.

**Upload your textbook or photo → Ask questions → Get intelligent, context-aware answers in your language.**

---

## Features

- **AI Chat** — Ask any academic question, get structured answers with real-world examples
- **RAG (Retrieval-Augmented Generation)** — Upload PDFs, DOCX, TXT, or images and the AI uses your curriculum as context
- **Image Understanding** — Upload diagrams, screenshots, or photos of notes; Gemini Vision extracts and indexes the content
- **Voice Lessons (Tutor Mode)** — Generate full spoken lessons brick-by-brick with Edge TTS neural voices
- **Multilingual Support** — English, Hindi, Hinglish, Urdu, Persian, German, French
- **Follow-up Questions** — Ask questions mid-lesson and get spoken answers without losing your place
- **Progress Tracker** — Live counters, topic frequency charts, session history
- **Documents Tab** — Drag-and-drop upload with indexing status
- **Beautiful UI** — Dark academic theme, smooth animations, responsive design

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND                            │
│         index.html (HTML + CSS + Vanilla JS)            │
│   Chat Tab │ Tutor Tab │ Progress Tab │ Documents Tab   │
└─────────────────────┬───────────────────────────────────┘
                      │  HTTP (localhost:8000)
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    BACKEND                              │
│              FastAPI (Python)                           │
│  /api/chat  /api/upload  /api/tutor  /api/tts  /api/docs│
└──────────┬──────────────────────┬────────────┬─────────-┘
           │                      │            │
           ▼                      ▼            ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  llm_handler.py  │  │  rag_handler.py  │  │   edge_tts       │
│  Groq LLaMA 3.3  │  │  ChromaDB +      │  │  Neural TTS      │
│  (Chat + Lessons │  │  MiniLM-L6-v2 +  │  │  (7 languages)   │
│  + Follow-ups)   │  │  Gemini Vision   │  │                  │
└──────────────────┘  └─────────────────┘  └──────────────────┘
```

**Data Flow:**
1. User types question (or requests a lesson) in the frontend
2. Frontend sends POST to FastAPI backend
3. **Chat:** If RAG is ON, ChromaDB retrieves relevant chunks from uploaded docs/images
4. Context + question sent to **Groq LLaMA 3.3 70B** for a response
5. **Tutor Mode:** Groq generates a structured 6-brick lesson as JSON, streamed brick-by-brick
6. **TTS:** Each brick is sent to `/api/tts` → Edge TTS returns audio streamed to the browser
7. **Image upload:** Gemini Vision extracts educational content, which is indexed in ChromaDB

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Backend runtime | [python.org](https://python.org) |
| Groq API Key | Run LLaMA 3.3 70B (free tier) | [console.groq.com](https://console.groq.com) |
| Gemini API Key | Image content extraction | [aistudio.google.com](https://aistudio.google.com) |

> Both Groq and Gemini offer generous **free tiers** — no credit card needed to get started.

---

## Installation & Setup

### Step 1 — Clone / Download This Project

```bash
git clone https://github.com/mohdanaslko/edu-tutor.git
cd edu-tutor
```

### Step 2 — Create a `.env` File

Inside the `backend/` folder, create a `.env` file:

```env
Groq_API=your_groq_api_key_here
GEMINI_API=your_gemini_api_key_here
```

**Get your keys (both free):**
- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Step 3 — Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> The first run will download the `all-MiniLM-L6-v2` embedding model (~90MB). This only happens once.

### Step 4 — Start the Backend

```bash
cd backend
python main.py
```

You should see:
```
✅ LLMHandler ready — Groq llama-3.3-70b (chat + lessons + followups). TTS via browser.
✅ RAG Handler initialized with ChromaDB + SentenceTransformer + Gemini Vision.
📡 API available at: http://localhost:8000
📖 API docs at:      http://localhost:8000/docs
```

### Step 5 — Open the Frontend

```bash
# macOS
open frontend/index.html

# Windows
start frontend/index.html

# Linux
xdg-open frontend/index.html
```

> No build step, no npm, no webpack. Just open the file.

---

## How to Use

### Asking Questions
1. Type your question in the chat input
2. Press **Enter** to send (Shift+Enter for new line)
3. Toggle **"Use Docs"** in the header to include your uploaded documents as context

### Uploading Documents & Images
- Click the **upload button** in the chat, OR go to the **Documents tab**
- Drag-and-drop your files — PDFs, DOCX, TXT, or images (PNG/JPG/WEBP/GIF)
- Images are processed by **Gemini Vision** to extract text, formulas, diagrams, and tables

### Using Tutor Mode (Voice Lessons)
1. Open the **Tutor** tab
2. Enter any topic and select your language
3. Click **Start Lesson** — a 6-part spoken lesson is generated and read aloud
4. Navigate bricks with Prev/Next, or ask follow-up questions at any time

### Supported Languages
English · Hindi · Hinglish · Urdu · Persian · German · French

### Viewing Progress
Click **Progress** in the sidebar to see session stats, topic distribution, and recent questions.

---

## Project Structure

```
edu-tutor/
├── backend/
│   ├── main.py           # FastAPI routes & server
│   ├── llm_handler.py    # Groq / LLaMA 3.3 interface (chat, lessons, follow-ups)
│   ├── rag_handler.py    # ChromaDB vector store + Gemini Vision for images
│   ├── .env              # API keys (Groq + Gemini) — not committed to git
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── index.html        # Complete UI (HTML + CSS + JS)
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check backend status and model info |
| POST | `/api/chat` | Send a message, get AI response (with optional RAG) |
| POST | `/api/upload` | Upload a document or image for RAG indexing |
| GET | `/api/documents` | List all indexed documents |
| POST | `/api/tutor/start` | Generate a structured spoken lesson |
| POST | `/api/tutor/followup` | Answer a student's mid-lesson question |
| POST | `/api/tts` | Convert text to speech (returns MP3 stream) |

### Chat Request
```json
{
  "message": "Explain Newton's second law",
  "use_rag": true
}
```

### Tutor Start Request
```json
{
  "topic": "Photosynthesis",
  "language": "Hindi"
}
```

### TTS Request
```json
{
  "text": "Welcome to today's lesson on photosynthesis.",
  "language": "English"
}
```

---

## Customization

### Change the Chat Model

In `backend/llm_handler.py`:
```python
# Other free Groq models:
# "llama-3.1-70b-versatile"  - slightly older, equally capable
# "llama-3.1-8b-instant"     - ultra-fast, lower quality
# "mixtral-8x7b-32768"       - long context window
self.llm = ChatGroq(model="llama-3.3-70b-versatile", ...)
```

### Change the TTS Voice

In `backend/main.py`, update the `voice_map` dictionary:
```python
voice_map = {
    "English": "en-GB-SoniaNeural",   # British female
    "Hindi":   "hi-IN-MadhurNeural",  # Hindi male
    ...
}
```
Browse all available voices: [learn.microsoft.com/azure/ai-services/speech/language-support](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support)

### Persist Documents Across Restarts

In `backend/rag_handler.py`:
```python
# From (in-memory, resets on restart):
self.client = chromadb.Client()

# To (persistent, survives restarts):
self.client = chromadb.PersistentClient(path="./chroma_db")
```

### Change Backend Port

In `backend/main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
```
And update the API base URL in `frontend/index.html`:
```js
const API = 'http://localhost:8080';
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Groq_API key is missing` | Create `backend/.env` and add your Groq key |
| `GEMINI_API` errors on image upload | Add your Gemini API key to `backend/.env` |
| Upload fails for images | Check that your file is PNG/JPG/WEBP/GIF and backend is running |
| No audio in Tutor mode | Ensure your browser allows audio autoplay; check network tab for TTS errors |
| Slow first lesson | Normal — LLaMA 3.3 70B generates all 6 bricks in one call; ~5–10s on Groq free tier |
| Port 8000 in use | Change port in `main.py` and `index.html` |
| `Backend not running` toast | Run `python main.py` from the `backend/` folder |

---

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| AI Chat & Lessons | LLaMA 3.3 70B via Groq API | Free tier |
| Image Understanding | Gemini 2.0 Flash Vision | Free tier |
| Text Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Free |
| Vector Database | ChromaDB (local) | Free |
| Text-to-Speech | Microsoft Edge TTS (Neural voices) | Free |
| Backend | FastAPI + Uvicorn | Free |
| LLM Framework | LangChain + LangChain-Groq | Free |
| Frontend | Vanilla HTML/CSS/JS | Free |

**Total monthly cost: ₹0**

---

## License

MIT License — use, modify, and share freely.

---

*Built with love for students who deserve better learning tools.*