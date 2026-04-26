import pandas as pd
import csv
import time
import os
from dotenv import load_dotenv

from gemma_wrapper import get_gemma_response
from llama_wrapper import get_llama_response
from gpt_wrapper import get_openai_response

load_dotenv()

GEMMA_KEY = os.getenv("GEMMA_KEY")
LLAMA_KEY = os.getenv("LLAMA_KEY")
GPT_KEY = os.getenv("GPT_KEY")

def build_prompt(clue, ans):
    # prompt taken from Sadallah (2025) paper
    return f"""You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue.
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

    - assemblage: The answer is broken up into smaller parts and each syllable or part is given a separate clue. These separate clues are then put together into one clue. 
        Example: Brash gets a Prime Minister employment, but it’s drudgery (6,4) 
        The answer: Donkey work

    Only output the wordplay type. 
    Clue: {clue} 
    The answer: {ans} 
    Output:"""

def run_classification(input_csv, output_csv, model_choice):
    df = pd.read_csv(input_csv)
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(list(df.columns) + [f"{model_choice}_predicted"])

        for _, row in df.iterrows():
            print(f"[{model_choice}] Current Index: {row['Index']}")
            prompt = build_prompt(row['Clue'], row['Solution'])
            
            if model_choice == "gemma":
                prediction = get_gemma_response(prompt, GEMMA_KEY)
            elif model_choice == "llama":
                prediction = get_llama_response(prompt, LLAMA_KEY)
            elif model_choice == "gpt":
                prediction = get_gpt_response(prompt, GPT_KEY)
            
            clean_prediction = prediction.strip().lower()

            writer.writerow(list(row), + [clean_prediction])
            f.flush()

            time.sleep(1)
            
    print(f"Results saved to {output_csv}")

# comment out to run one by one
# run_classification("annotated_clues.csv", "results_gemma.csv", "gemma")
# run_classification("annotated_clues.csv", "results_llama.csv", "llama")
# run_classification("annotated_clues.csv", "results_gpt.csv", "openai")