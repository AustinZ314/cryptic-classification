import pandas as pd
import csv
import time
import os
import json
from dotenv import load_dotenv

from gemma_wrapper import get_gemma_response
from llama_wrapper import get_llama_response
from gpt_wrapper import get_gpt_response

load_dotenv()

GEMMA_KEY = os.getenv("GEMMA_KEY")
LLAMA_KEY = os.getenv("LLAMA_KEY")
GPT_KEY = os.getenv("GPT_KEY")

def build_batch_prompt(clue_chunk):
    prompt = """You are a cryptic crosswords expert. I will give you a list of clues. 
For each clue, please extract the wordplay type.
Here is a list of all possible wordplay types, and their descriptions: 
- anagram: An anagram is a word (or words) that, when rearranged, forms a different word or phrase. 
    Example: Ms Reagan is upset by the executives (8)
    The answer: Managers

- hidden word: The answer is found in the clue itself, amongst other words. 
    Example: Confront them in the tobacco store (6)
    The answer: Accost

- double definition: Clues contain two meanings of the same word. The words may be pronounced differently, but must be spelt the same. 
    Example: Footwear for pack animals (5)
    The answer: Mules

- container: One word is placed inside another (or outside another) to get the answer. 
    Example: Curse about the Maori jumper (7)
    The answer: Sweater

- assemblage: The answer is broken up into smaller parts and each syllable or part is given a separate clue. 
    Example: Brash gets a Prime Minister employment, but it’s drudgery (6,4)
    The answer: Donkey work

Output ONLY a JSON list of objects, each with "id" and "type". Do not include any other text.
Example Output: [{"id": 1, "type": "anagram"}, {"id": 2, "type": "container"}]

Clues to classify:"""

    for item in clue_chunk:
        prompt += f"\nID: {item['Index']} | Clue: {item['Clue']} | Answer: {item['Solution']}"
    
    prompt += "\nOutput:"
    return prompt

def run_batch_classification(input_csv, output_csv, model_choice, sleep_time=5, batch_size=32):
    df = pd.read_csv(input_csv)
    file_exists = os.path.isfile(output_csv)
    
    # Open output file in append mode to save results incrementally
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Reconstruct the header using original columns + the prediction column
            writer.writerow(list(df.columns) + [f"{model_choice}_predicted"])

        # Iterate through the dataframe in chunks of batch_size
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            current_indices = batch_df['Index'].tolist()
            
            print(f"[{model_choice}] Processing Batch: Indices {current_indices[0]} to {current_indices[-1]}")
            
            # Convert the batch to a list of dicts for the prompt builder
            clue_chunk = batch_df[['Index', 'Clue', 'Solution']].to_dict('records')
            prompt = build_batch_prompt(clue_chunk)
            
            # Call the appropriate model wrapper
            if model_choice == "gemma":
                # Uses Gemma-4-31B-it as discussed
                response = get_gemma_response(prompt, GEMMA_KEY)
            elif model_choice == "llama":
                # Uses Llama-3.1-8B-Instruct (via Groq)
                response = get_llama_response(prompt, LLAMA_KEY)
            elif model_choice == "gpt":
                # Uses GPT-4o-mini
                response = get_gpt_response(prompt, GPT_KEY)
            
            try:
                if "empty" in response:
                    print(f"Skipping JSON parse for batch {i} due to API error: {response}")
                    for _, row_data in batch_df.iterrows():
                        writer.writerow(list(row_data) + ["Error: Empty response"])
                    f.flush()
                    # time.sleep(sleep_time)
                    continue
                # Parse the JSON response
                # We strip potential markdown blocks like ```json ... ```
                clean_json = response.strip().replace("```json", "").replace("```", "")
                predictions = json.loads(clean_json)
                
                # Write each prediction from the batch into the CSV
                for pred in predictions:
                    # Match the predicted ID back to the original row in the current batch
                    row_data = batch_df[batch_df['Index'] == pred['id']].iloc[0]
                    writer.writerow(list(row_data) + [pred['type'].strip().lower()])
                
                f.flush() # Force write to disk so we don't lose data on crash
                
            except Exception as e:
                print(f"Error parsing JSON for batch starting at index {i}: {e}")
                print(f"Raw response was: {response}")

            # Sleep to respect rate limits (adjust based on model requirements)
            # time.sleep(sleep_time)
            
    print(f"[{model_choice}] Completed. Results saved to {output_csv}")

# --- EXECUTION ---
# run_batch_classification("gemma_missing.csv", "results_gemma.csv", "gemma", 0)
run_batch_classification("hand_annotated_500.csv", "results_llama.csv", "llama", 0)
# run_batch_classification("annotated_clues.csv", "results_gpt.csv", "gpt", 0)