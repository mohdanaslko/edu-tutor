import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API = os.getenv("Groq_API")

LANG_INSTRUCTIONS = {
    "English":  "Clear, simple, warm English. Like a friendly patient teacher. Short sentences.",
    "Hindi":    "पूरी तरह हिंदी में। सरल, स्पष्ट और दोस्ताना। छोटे वाक्य।",
    "Hinglish": "Hindi+English mix. E.g. 'Aaj hum X seekhenge. This concept interesting hai kyunki...'",
    "Urdu":     "مکمل اردو میں۔ سادہ، واضح اور دوستانہ۔",
    "Persian":  "کاملاً فارسی. ساده و دوستانه.",
    "German":   "Vollständig Deutsch. Einfach und freundlich.",
    "French":   "Entièrement français. Simple et chaleureux.",
}


class LLMHandler:

    def __init__(self):
        if not GROQ_API:
            raise EnvironmentError("Groq_API key is missing. Set it as an environment variable in Render.")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API,
            temperature=0.7,
            max_tokens=2048
        )
        print("✅ LLMHandler ready — Groq llama-3.3-70b (chat + lessons + followups). TTS via browser.")

    # ── Chat ──────────────────────────────────────────────────────────────────

    def generate_response(self, prompt: str, context: str = "") -> str:
        context_block = ""
        if context.strip():
            context_block = (
                f"\n\n## Study Material:\n{context}\n\n"
                "Use the above material to answer. If not covered, use your own knowledge and say so.\n"
            )
        full_prompt = (
            "You are EduTutor — a brilliant, patient academic tutor. "
            "Explain step-by-step, give 2-3 relatable examples, use analogies, be warm. "
            "Structure your response with headings when helpful.\n"
            f"{context_block}\n## Student's Question:\n{prompt}"
        )
        return self.llm.invoke(full_prompt).content

    # ── Tutor lesson ──────────────────────────────────────────────────────────

    def generate_tutor_lesson(self, topic: str, language: str) -> dict:
        """
        Generate a full spoken lesson via Groq (free tier, no Gemini quota needed).
        TTS is handled entirely by the browser's speechSynthesis API.
        """
        lang_instruction = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["English"])

        prompt = (
            f'You are EduTutor. Create a warm spoken lesson on "{topic}" in {language}.\n'
            f"Style: {lang_instruction}\n\n"
            "Rules:\n"
            "- Speak directly to the student like a kind teacher sitting next to them.\n"
            "- Use short sentences (max 20 words each) so it sounds natural when spoken aloud.\n"
            "- Each brick teaches ONE idea using a simple real-life analogy.\n"
            "- Brick content must be 4-5 sentences. No bullet points. Pure flowing speech.\n"
            f"- Write exactly 6 bricks. Titles: 3-5 words in {language}.\n"
            "- NO markdown, NO asterisks, NO symbols — plain text only (it will be read aloud).\n"
            "- Return ONLY the JSON object below. Zero extra text before or after.\n\n"
            f'JSON to fill (all values in {language}, replace angle-bracket text):\n'
            '{\n'
            '  "topic": "<topic>",\n'
            '  "language": "<language>",\n'
            '  "welcome": "<warm 2-sentence welcome that names the topic and invites the student>",\n'
            '  "bricks": [\n'
            '    {"id": 1, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"},\n'
            '    {"id": 2, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"},\n'
            '    {"id": 3, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"},\n'
            '    {"id": 4, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"},\n'
            '    {"id": 5, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"},\n'
            '    {"id": 6, "title": "<title>", "content": "<4-5 spoken sentences, one idea, one analogy>"}\n'
            '  ],\n'
            '  "closing": "<1 warm encouraging sentence that celebrates finishing the lesson>"\n'
            '}'
        )

        raw = ""
        try:
            raw = self.llm.invoke(prompt).content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError(f"No JSON object found. Got: {raw[:300]}")
            raw = raw[start:end]
            lesson = json.loads(raw)
            missing = {"topic", "language", "welcome", "bricks", "closing"} - lesson.keys()
            if missing:
                raise ValueError(f"Missing fields: {missing}")
            if len(lesson.get("bricks", [])) != 6:
                raise ValueError(f"Expected 6 bricks, got {len(lesson.get('bricks', []))}")
            return lesson
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON from LLM: {e}\nRaw: {raw[:400]}")
        except Exception as e:
            raise Exception(f"Lesson generation failed: {e}")

    # ── Follow-up ─────────────────────────────────────────────────────────────

    def answer_followup(self, question: str, topic: str, language: str,
                        current_brick: int, total_bricks: int) -> dict:
        """
        Answer a student question mid-lesson via Groq.
        Returns {"answer": str} — browser speaks it via speechSynthesis.
        """
        lang_instruction = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["English"])
        prompt = (
            f'You are EduTutor. You are teaching "{topic}" in {language} '
            f"(currently on section {current_brick} of {total_bricks}).\n"
            f"Style: {lang_instruction}\n\n"
            "The student just asked a question. Answer it in 3-4 short, warm spoken sentences. "
            "No bullet points. Plain text only (it will be read aloud). "
            f"End with one encouraging phrase to resume the lesson in {language}.\n\n"
            f"Student's question: {question}"
        )
        try:
            answer = self.llm.invoke(prompt).content.strip()
            return {"answer": answer, "audio_b64": ""}
        except Exception as e:
            raise Exception(f"Follow-up failed: {e}")