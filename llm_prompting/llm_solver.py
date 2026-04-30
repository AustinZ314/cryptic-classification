import pandas as pd
import csv
import time
import os
import re
from dotenv import load_dotenv
from gemma_wrapper import get_gemma_response
from llama_wrapper import get_llama_response
from gpt_wrapper import get_gpt_response

load_dotenv()

# calculate solution length to append to clue
def get_soln_length(solution):
    if not solution or not isinstance(solution, str):
        return ""
    
    parts = re.split(r'([ \-])', solution.strip())

    length_parts = []
    for part in parts:
        if part == ' ':
            length_parts.append(",")
        elif part == '-':
            length_parts.append("-")
        else:
            length_parts.append(str(len(part)))
        
    return f"({''.join(length_parts)})"    

def build_solving_prompt(clue_info):
    prompt = """You are a cryptic crosswords expert.
A cryptic clue consists of a definition and a wordplay.
The definition is a synonym of the answer and usually comes at the beginning or the end of the clue.
The wordplay gives some instructions on how to get to the answer in another (less literal) way.
The number(s) in the parentheses at the end of the clue indicates the number of letters in the answer.
Extract the definiton and the wordplay in each clue, and use them to solve the clues.
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

Output only the solution.
Clue: """

    length_str = get_soln_length(clue_info.get('Spaced_Solution', ''))
    prompt += f"{clue_info['Clue']} {length_str}"
    
    prompt += "\nOutput:"
    return prompt

def run_solving(input_csv, output_csv, model_choice):
    df = pd.read_csv(input_csv)
    
    processed_indices = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_indices = set(existing_df['Index'].tolist())
    
    df_to_process = df[~df['Index'].isin(processed_indices)]
    
    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(list(df.columns) + [f"{model_choice}_solved_answer"])

        for index, row in df_to_process.iterrows():
            current_index = row['Index']
            
            print(f"[{model_choice}] Solving clue with index {current_index}")
            
            prompt = build_solving_prompt(row)
            
            if model_choice == "gemma":
                response = get_gemma_response(prompt, os.getenv("GEMMA_KEY"))
            elif model_choice == "llama":
                response = get_llama_response(prompt, os.getenv("LLAMA_KEY"))
            elif model_choice == "gpt":
                response = get_gpt_response(prompt, os.getenv("GPT_KEY"))
            
            try:
                model_ans = response.strip().upper()
                ground_truth = str(row['Solution']).strip().upper()

                clean_model = "".join(filter(str.isalpha, model_ans))
                clean_truth = "".join(filter(str.isalpha, ground_truth))
                    
                is_correct = 1 if clean_model == clean_truth else 0
                writer.writerow(row.tolist() + [model_ans, is_correct])
                f.flush()
            except Exception as e:
                print(f"Error at index {current_index}: {e}")

# change input clue type manually
run_solving("./categories/container.csv", "./solved/gemma_solved_container.csv", "gemma")
# run_solving("./categories/anagram.csv", "./solved/llama_solved_anagram.csv", "llama")
# run_solving("./categories/anagram.csv", "./solved/gpt_solved_anagram.csv", "gpt")