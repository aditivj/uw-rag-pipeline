from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(query: str, chunks: list, cache_hit: bool):
    context = "\n\n".join(chunks) if chunks else "No context found."
    prompt = f"""You are an insurance document assistant.
Answer the question using only the context provided.
If the answer is not in the context, say "I could not find this in the documents."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )

    answer = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return answer, input_tokens, output_tokens
