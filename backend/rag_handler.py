import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict
from dotenv import load_dotenv
from google import genai                
from PIL import Image

load_dotenv()

GEMINI_API = os.getenv("GEMINI_API")

# BUG FIX: original did `genai = google.genai.Client(api_key=GEMINI_API)` which
# overwrote the module-level `genai` name. Use a distinct variable name.
gemini_client = genai.Client(api_key=GEMINI_API)

GEMINI_MODEL     = "gemini-2.0-flash"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
DOC_EXTENSIONS   = {'.pdf', '.docx', '.doc', '.txt'}
ALL_EXTENSIONS   = IMAGE_EXTENSIONS | DOC_EXTENSIONS


class RAGHandler:

    def __init__(self):
        self.client = chromadb.Client()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="edu_tutor_knowledge_base",
            embedding_function=self.ef
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.document_metadata: Dict[str, dict] = {}
        print("✅ RAG Handler initialized with ChromaDB + SentenceTransformer + Gemini Vision.")

    # ─────────────────────────── IMAGE LOADING ────────────────────────────────

    def load_image(self, file_path: str, filename: str) -> int:
        """Use Gemini Vision to extract educational content from image, then index."""
        try:
            img = Image.open(file_path)

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

            # BUG FIX: original called `model.generate_content([prompt, img])` where
            # `model` was `genai.GenerativeModel(...)` — that's the OLD SDK.
            # New google-genai SDK: client.models.generate_content(model=..., contents=[...])
            from google.genai import types as gtypes
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    gtypes.Part.from_text(extraction_prompt),
                    gtypes.Part.from_bytes(
                        data=open(file_path, "rb").read(),
                        mime_type=f"image/{file_path.rsplit('.',1)[-1].lower().replace('jpg','jpeg')}"
                    )
                ]
            )
            extracted_text = response.text.strip()

            if not extracted_text:
                raise ValueError(f"No content extracted from '{filename}'")

            doc = Document(
                page_content=extracted_text,
                metadata={"source": filename, "type": "image"}
            )
            chunks = self.text_splitter.split_documents([doc])
            if not chunks:
                chunks = [doc]

            ids, texts, metadatas = [], [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{filename}_chunk_{i}")
                texts.append(chunk.page_content)
                metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "type": "image"
                })

            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            self.document_metadata[filename] = {"chunks": len(chunks), "type": "image"}
            print(f"✅ Image '{filename}' indexed: {len(chunks)} chunk(s)")
            return len(chunks)

        except Exception as e:
            raise Exception(f"Could not process image '{filename}': {str(e)}")

    # ──────────────────────────── DOCUMENT LOADING ────────────────────────────

    def load_document(self, file_path: str, filename: str) -> int:
        """Load PDF / DOCX / TXT, split into chunks and index in ChromaDB."""
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
            raise Exception(f"Could not load '{filename}': {str(e)}")

        if not document:
            raise ValueError(f"No content found in '{filename}'")

        chunks = self.text_splitter.split_documents(document)
        if not chunks:
            raise ValueError(f"Failed to split '{filename}' into chunks.")

        ids, texts, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}_chunk_{i}")
            texts.append(chunk.page_content)
            metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "document"
            })

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        self.document_metadata[filename] = {"chunks": len(chunks), "type": "document"}
        print(f"✅ Document '{filename}' indexed: {len(chunks)} chunk(s)")
        return len(chunks)

    # ─────────────────────────────── QUERY ────────────────────────────────────

    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        total_docs = self.collection.count()
        if total_docs == 0:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_docs)
            )
            if not results['documents'] or not results['documents'][0]:
                return []
            output = []
            for text, meta in zip(results['documents'][0], results['metadatas'][0]):
                if text.strip():
                    output.append({
                        "text": text,
                        "source": meta.get("source", "unknown"),
                        "type": meta.get("type", "document")
                    })
            return output
        except Exception as e:
            print(f"Query error: {str(e)}")
            return []

    # ─────────────────────────── DOCUMENT LIST ────────────────────────────────

    def get_document_list(self) -> List[Dict]:
        return [
            {
                "filename": name,
                "chunks": meta["chunks"],
                "type": meta.get("type", "document")
            }
            for name, meta in self.document_metadata.items()
        ]