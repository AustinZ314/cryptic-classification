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

def run_batch_classification(input_csv, output_csv, model_choice, batch_size=32):
    df = pd.read_csv(input_csv)

    # logic to resume running if it crashes
    processed_indices = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_indices = set(existing_df['Index'].tolist())
    
    # filter out rows that have already been classified
    df_to_process = df[~df['Index'].isin(processed_indices)]
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(list(df.columns) + [f"{model_choice}_predicted"])

        for i in range(0, len(df_to_process), batch_size):
            batch_df = df_to_process.iloc[i : i + batch_size]
            current_indices = batch_df['Index'].tolist()
            
            print(f"[{model_choice}] Processing Batch: Indices {current_indices[0]} to {current_indices[-1]}")
            
            clue_chunk = batch_df[['Index', 'Clue', 'Solution']].to_dict('records')
            prompt = build_batch_prompt(clue_chunk)
            
            if model_choice == "gemma":
                response = get_gemma_response(prompt, GEMMA_KEY)
            elif model_choice == "llama":
                response = get_llama_response(prompt, LLAMA_KEY)
            elif model_choice == "gpt":
                response = get_gpt_response(prompt, GPT_KEY)
            
            if "Empty" in response:
                print(f"Skipping batch starting at {current_indices[0]} due to API error: {response}")
                for _, row_data in batch_df.iterrows():
                    writer.writerow(list(row_data) + ["Error: Empty response"])
                f.flush()
                continue

            try:
                clean_json = response.strip().replace("```json", "").replace("```", "")
                predictions = json.loads(clean_json)
                
                for pred in predictions:
                    match = batch_df[batch_df['Index'] == pred['id']]
                    if not match.empty:
                        row_data = match.iloc[0]
                        writer.writerow(list(row_data) + [pred['type'].strip().lower()])
                
                f.flush()
                
            except Exception as e:
                print(f"Error parsing JSON for batch starting at index {i}: {e}")
                print(f"Raw response was: {response}")
            
    print(f"[{model_choice}] Completed. Results saved to {output_csv}")

# --- EXECUTION ---
run_batch_classification("austin_classify.csv", "classified_dataset.csv", "gemma")
# run_batch_classification("hand_annotated_500.csv", "results_llama.csv", "llama")
# run_batch_classification("hand_annotated_500.csv", "results_gpt.csv", "gpt")