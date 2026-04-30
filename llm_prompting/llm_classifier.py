import pandas as pd
import csv
import time
import os
from dotenv import load_dotenv

from gemma_wrapper import get_gemma_response
from llama_wrapper import get_llama_response
from gpt_wrapper import get_gpt_response

load_dotenv()

GEMMA_KEY = os.getenv("GEMMA_KEY")
LLAMA_KEY = os.getenv("LLAMA_KEY")
GPT_KEY = os.getenv("GPT_KEY")

def build_prompt(clue_info):
    prompt = """You are a cryptic crosswords expert. I will give you a clue and its solution. 
As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue. 
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

Only output the wordplay type.
Clue: """

    prompt += f"{clue_info['Clue']}\nAnswer: {clue_info['Solution']}"
    
    prompt += "\nOutput:"
    return prompt

def run_classification(input_csv, output_csv, model_choice):
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

        for index, row in df_to_process.iterrows():
            current_index = row['Index']
            
            print(f"[{model_choice}] Classifying clue with index {current_index}")
        
            prompt = build_prompt(row)
            
            if model_choice == "gemma":
                response = get_gemma_response(prompt, GEMMA_KEY)
            elif model_choice == "llama":
                response = get_llama_response(prompt, LLAMA_KEY)
            elif model_choice == "gpt":
                response = get_gpt_response(prompt, GPT_KEY)
            
            if "Empty" in response:
                print(f"Skipping index {current_index} due to API error: {response}")
                continue

            try:
                clean_res = response.strip().lower()
                writer.writerow(row.tolist() + [clean_res])
                f.flush()
                time.sleep(2) # change depending on model to avoid rate limits

            except Exception as e:
                print(f"Error parsing response for clue at index {current_index}: {e}")
            
    print(f"[{model_choice}] Completed. Results saved to {output_csv}")

# --- EXECUTION ---
run_classification("full_dataset.csv", "classified_dataset.csv", "gemma")
# run_classification("hand_annotated_500.csv", "results_llama.csv", "llama")
# run_classification("hand_annotated_500.csv", "results_gpt.csv", "gpt")