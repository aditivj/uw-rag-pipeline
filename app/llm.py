from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# def generate_answer(query: str, chunks: list, cache_hit: bool):
#     context = "\n\n".join(chunks) if chunks else "No context found."
#     prompt = f"""You are an insurance document assistant.
# Answer the question using only the context provided.
# If the answer is not in the context, say "I could not find this in the documents."

# Context:
# {context}

# Question: {query}

# Answer:"""

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=512
#     )

#     answer = response.choices[0].message.content
#     input_tokens = response.usage.prompt_tokens
#     output_tokens = response.usage.completion_tokens

#     return answer, input_tokens, output_tokens

def generate_answer(query: str, chunks: list, cache_hit: bool):
    context = "\n\n".join(chunks) if chunks else "No context found."
    prompt = f"""You are an insurance document assistant. Answer the question using only the context provided.

Rules:
- If a context chunk is tagged with a section heading like [SECTION III — EXCLUSIONS], 
  treat any item listed there as NOT covered / excluded.
- If a context chunk is tagged with a heading like [COVERAGE] or appears in a coverage table, 
  treat it as covered, and state the limit if given.
- Give a direct, confident answer (Yes/No or a specific value) whenever the context contains 
  enough information to determine it — do not hedge if the section heading already answers the question.
- Only say "I could not find this in the documents" if the topic is genuinely absent from the context.

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


