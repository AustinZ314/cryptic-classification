from google import genai

def get_gemma_response(prompt, api_key):
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemma-4-31b-it', 
            contents=prompt
        )

        if response and response.text:
            return response.text.strip().lower()
        else:
            return "Error: Empty response from Gemma"
    except Exception as e:
        return f"Error: {e}"