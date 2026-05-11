import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from PIL import Image

load_dotenv()

GEMINI_API      = os.getenv("GEMINI_API")
GEMINI_MODEL    = "gemini-2.0-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

gemini_client = genai.Client(api_key=GEMINI_API) if GEMINI_API else None

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
DOC_EXTENSIONS   = {'.pdf', '.docx', '.doc', '.txt'}
ALL_EXTENSIONS   = IMAGE_EXTENSIONS | DOC_EXTENSIONS


# ── FIX 2a: ChromaDB embedding function ──────────────────────────────────────
# ChromaDB's EmbeddingFunction protocol requires the callable to accept
# `input: Documents` (a list of strings).  The `name()` method must exist but
# ChromaDB also checks for it as an attribute via hasattr — making it a plain
# method is fine, but the class must inherit from
# chromadb.EmbeddingFunction to satisfy the type-check in newer versions.

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    ChromaDB-compatible embedding function backed by Gemini embedding model.
    Inherits from chromadb.EmbeddingFunction so collection validation passes.
    """

    def __init__(self, client: genai.Client, task_type: str = "RETRIEVAL_DOCUMENT"):
        self._client    = client
        self._task_type = task_type

    # ChromaDB calls this to identify the function stored in the collection
    def name(self) -> str:
        return f"gemini-{EMBEDDING_MODEL}-{self._task_type}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a batch of texts and return float vectors."""
        if not input:
            return []
        response = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=input,
            config=gtypes.EmbedContentConfig(task_type=self._task_type),
        )
        return [e.values for e in response.embeddings]


_ef_index = GeminiEmbeddingFunction(gemini_client, task_type="RETRIEVAL_DOCUMENT") if gemini_client else None
_ef_query = GeminiEmbeddingFunction(gemini_client, task_type="RETRIEVAL_QUERY")    if gemini_client else None


class RAGHandler:

    def __init__(self):
        # ── FIX 2b: Use /tmp on Render (ephemeral but survives the request).
        # For true persistence across deploys, attach a Render Disk and set
        # CHROMA_DB_PATH=/var/data/chroma_db in your environment variables.
        chroma_path = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
        os.makedirs(chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(
            name="edu_tutor_knowledge_base",
            embedding_function=_ef_index,
            metadata={"hnsw:space": "cosine"},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.document_metadata: Dict[str, dict] = {}
        print(
            f"✅ RAG Handler ready — ChromaDB @ {chroma_path} "
            f"+ Gemini {EMBEDDING_MODEL} embeddings + Gemini Vision."
        )

    # ─────────────────────────── IMAGE LOADING ────────────────────────────────

    def load_image(self, file_path: str, filename: str) -> int:
        """Use Gemini Vision to extract educational content, then index."""
        if not gemini_client:
            raise EnvironmentError("GEMINI_API key is missing — cannot process images.")
        try:
            Image.open(file_path)   # validate before sending

            extraction_prompt = (
                "You are an expert educational content extractor.\n"
                "Extract everything useful for a student from this image:\n"
                "1. ALL visible text — copy exactly\n"
                "2. Math/science formulas — write clearly\n"
                "3. Diagrams, graphs, charts — describe every detail (labels, axes, values)\n"
                "4. Tables — reproduce full structure\n"
                "5. Annotations, captions, footnotes\n\n"
                "Format as clear educational notes. Be thorough."
            )

            ext      = file_path.rsplit('.', 1)[-1].lower().replace('jpg', 'jpeg')
            mime     = f"image/{ext}"
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    gtypes.Part.from_text(text=extraction_prompt),
                    gtypes.Part.from_bytes(
                        data=open(file_path, "rb").read(),
                        mime_type=mime,
                    ),
                ],
            )
            extracted_text = response.text.strip()

            if not extracted_text:
                raise ValueError(f"No content extracted from '{filename}'")

            doc    = Document(page_content=extracted_text, metadata={"source": filename, "type": "image"})
            chunks = self.text_splitter.split_documents([doc]) or [doc]
            return self._index_chunks(chunks, filename, doc_type="image")

        except Exception as e:
            raise Exception(f"Could not process image '{filename}': {e}")

    # ──────────────────────────── DOCUMENT LOADING ────────────────────────────

    def load_document(self, file_path: str, filename: str) -> int:
        """Load PDF / DOCX / TXT, split and index with Gemini embeddings."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ('.docx', '.doc'):
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            document = loader.load()
        except Exception as e:
            raise Exception(f"Could not load '{filename}': {e}")

        if not document:
            raise ValueError(f"No content found in '{filename}'")

        chunks = self.text_splitter.split_documents(document)
        if not chunks:
            raise ValueError(f"Failed to split '{filename}' into chunks.")

        return self._index_chunks(chunks, filename, doc_type="document")

    # ─────────────────────────── SHARED INDEXING ──────────────────────────────

    def _index_chunks(self, chunks: List[Document], filename: str, doc_type: str) -> int:
        ids, texts, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}_chunk_{i}")
            texts.append(chunk.page_content)
            metadatas.append({
                "source":       filename,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "type":         doc_type,
            })

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        self.document_metadata[filename] = {"chunks": len(chunks), "type": doc_type}
        print(f"✅ '{filename}' indexed: {len(chunks)} chunk(s) via Gemini {EMBEDDING_MODEL}")
        return len(chunks)

    # ─────────────────────────────── QUERY ────────────────────────────────────

    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        total_docs = self.collection.count()
        if total_docs == 0:
            return []
        try:
            query_embedding = _ef_query([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total_docs),
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            return [
                {
                    "text":   text,
                    "source": meta.get("source", "unknown"),
                    "type":   meta.get("type", "document"),
                }
                for text, meta in zip(results['documents'][0], results['metadatas'][0])
                if text.strip()
            ]
        except Exception as e:
            print(f"Query error: {e}")
            return []

    # ─────────────────────────── DOCUMENT LIST ────────────────────────────────

    def get_document_list(self) -> List[Dict]:
        return [
            {"filename": name, "chunks": meta["chunks"], "type": meta.get("type", "document")}
            for name, meta in self.document_metadata.items()
        ]