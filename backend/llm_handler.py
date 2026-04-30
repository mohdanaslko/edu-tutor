from langchain_ollama import ChatOllama
class LLMHandler:

    def __init__(self):
        self.llm = ChatOllama(model = 'mistral',
                              temperature = 0.7,
                              max_tokens = 1024)
        print(f"LLM Handler initialized {self.llm.model} model via Ollama.")

    def generate_response(self, prompt:str , context:str = " " )->str:
        """
        will take the context from document provided by the user via input
        """
        context_block = " "
        if context.strip():
            context_block = f"""{context} Use the above material to answer accurately. If the question is not covered in the material,
answer from your own knowledge but mention it."""

        full_prompt = f"""You are EduTutor — a brilliant, patient, and encouraging academic tutor.
Your job is to make complex topics simple and memorable.
 
## Your Teaching Style:
- Explain step by step, building from basics
- Always give 2-3 real-world examples that students can relate to
- Use analogies to make abstract ideas concrete
- Be warm, encouraging, and supportive
- Structure your answer clearly with headings when needed
- If you don't know something, say so honestly
{context_block}
## Student's Question:
{prompt}"""
        response = self.llm.invoke(full_prompt)
        return response.content
        