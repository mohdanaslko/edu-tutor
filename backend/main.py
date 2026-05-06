import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import edge_tts
from fastapi.responses import StreamingResponse
from backend.llm_handler import LLMHandler
from backend.rag_handler import RAGHandler, IMAGE_EXTENSIONS, DOC_EXTENSIONS

# ── Globals populated during lifespan startup, NOT at import time ─────────────
llm: LLMHandler = None
rag: RAGHandler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs AFTER the server binds its port — safe place for heavy init."""
    global llm, rag
    llm = LLMHandler()
    rag = RAGHandler()
    yield

app = FastAPI(title="EduTutor API", version="5.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
IMAGES_DIR   = os.path.join(FRONTEND_DIR, "images")
os.makedirs(FRONTEND_DIR, exist_ok=True)

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []

class TutorStartRequest(BaseModel):
    topic: str
    language: str = "English"

class TutorFollowupRequest(BaseModel):
    question: str
    topic: str
    language: str = "English"
    current_brick: int = 1
    total_bricks: int = 6

class TTSRequest(BaseModel):
    text: str
    language: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>EduTutor API is running. Frontend not found.</h2>", status_code=200)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "EduTutor backend is running!",
        "models": {
            "chat":    "llama-3.3-70b-versatile (Groq)",
            "lessons": "llama-3.3-70b-versatile (Groq)",
            "tts":     "edge-tts (server-side)"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        context, sources = "", []
        if req.use_rag:
            results = rag.query(req.message)
            if results:
                context = "\n\n---\n\n".join([r["text"] for r in results])
                seen = set()
                for r in results:
                    src = r["source"]
                    if src not in seen:
                        sources.append(src); seen.add(src)
        response = llm.generate_response(req.message, context)
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    all_allowed = DOC_EXTENSIONS | IMAGE_EXTENSIONS
    if ext not in all_allowed:
        raise HTTPException(status_code=400, detail=f"'{ext}' not supported.")
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content); tmp_path = tmp.name
        is_image = ext in IMAGE_EXTENSIONS
        chunks = rag.load_image(tmp_path, file.filename) if is_image else rag.load_document(tmp_path, file.filename)
        os.unlink(tmp_path)
        return {"message": f"'{file.filename}' indexed.", "chunks": chunks,
                "filename": file.filename, "type": "image" if is_image else "document"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def generate_speech(req: TTSRequest):
    voice_map = {
        "English":  "en-US-AriaNeural",
        "Hindi":    "hi-IN-SwaraNeural",
        "Hinglish": "hi-IN-SwaraNeural",
        "Urdu":     "ur-PK-UzmaNeural",
        "Persian":  "fa-IR-DilaraNeural",
        "German":   "de-DE-AmalaNeural",
        "French":   "fr-FR-DeniseNeural"
    }
    voice = voice_map.get(req.language, "en-US-AriaNeural")
    try:
        communicate = edge_tts.Communicate(req.text, voice)
        async def audio_stream():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tutor/start")
async def tutor_start(req: TutorStartRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    try:
        lesson = llm.generate_tutor_lesson(req.topic.strip(), req.language)
        return lesson
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tutor/followup")
async def tutor_followup(req: TutorFollowupRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = llm.answer_followup(
            req.question.strip(), req.topic, req.language,
            req.current_brick, req.total_bricks
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def get_documents():
    return {"documents": rag.get_document_list()}