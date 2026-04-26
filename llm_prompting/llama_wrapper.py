from groq import Groq

def get_llama_response(prompt, api_key):
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        content = chat_completion.choices[0].message.content
        if content:
            return content.strip().lower()
        return "Error: Empty response from Llama"
    except Exception as e:
        return f"Error: {e}"