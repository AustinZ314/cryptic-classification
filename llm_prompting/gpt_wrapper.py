from openai import OpenAI

def get_gpt_response(prompt, api_key):
    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content.strip().lower()
    except Exception as e:
        return f"Error: {e}"