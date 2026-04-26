import pandas as pd
import csv
import time
import os
import json
import re
from gemma_wrapper import get_gemma_response
from llama_wrapper import get_llama_response
from gpt_wrapper import get_gpt_response

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

def build_solving_prompt(clue_chunk):
    prompt = """You are a cryptic crosswords expert.
A cryptic clue consists of a definition and a wordplay.
The definition is a synonym of the answer and usually comes at the beginning or the end of the clue.
The wordplay gives some instructions on how to get to the answer in another (less literal) way.
The number/s in the parentheses at the end of the clue indicates the number of letters in the answer.
Extract the definiton and the wordplay in each clue, and use them to solve the clues.
Return ONLY a JSON list of objects, each with "id" and "answer".
Example Output: [{"id": 1, "answer": "MANGOS"}, {"id": 2, "answer": "ACCOST"}]

Clues to solve:"""

    for item in clue_chunk:
        length_str = get_soln_length(item.get('Spaced_Solution', ''))
        prompt += f"\nID: {item['Index']} | Clue: {item['Clue']} {length_str}"
    
    prompt += "\nOutput:"
    return prompt

def run_batch_solving(input_csv, output_csv, model_choice, category, batch_size=10):
    df = pd.read_csv(input_csv)
    
    processed_indices = set()
    if os.path.exists(output_csv) and os.stat(output_csv).st_size > 0:
        existing_df = pd.read_csv(output_csv)
        processed_indices = set(existing_df['Index'].tolist())
    
    df_to_process = df[~df['Index'].isin(processed_indices)]
    
    file_exists = os.path.exists(output_csv) and os.stat(output_csv).st_size > 0
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(list(df.columns) + [f"{model_choice}_solved_answer"])

        for i in range(0, len(df_to_process), batch_size):
            batch_df = df_to_process.iloc[i : i + batch_size]
            current_indices = batch_df['Index'].tolist()
            
            print(f"[{model_choice}] Solving Batch: {current_indices[0]} to {current_indices[-1]}")
            
            clue_chunk = batch_df[['Index', 'Clue', 'Spaced_Solution']].to_dict('records')
            prompt = build_solving_prompt(clue_chunk)
            
            if model_choice == "gemma":
                response = get_gemma_response(prompt, os.getenv("GEMMA_KEY"))
            elif model_choice == "llama":
                response = get_llama_response(prompt, os.getenv("LLAMA_KEY"))
            elif model_choice == "gpt":
                response = get_gpt_response(prompt, os.getenv("GPT_KEY"))
            
            try:
                clean_json = response.strip().replace("```json", "").replace("```", "")
                predictions = json.loads(clean_json)
                
                for pred in predictions:
                    match = batch_df[batch_df['Index'] == pred['id']]
                    if not match.empty:
                        row_data = match.iloc[0]
                        model_ans = pred['answer'].strip().upper()
                        ground_truth = str(row_data['Solution']).strip().upper()

                        clean_model = "".join(filter(str.isalpha, model_ans))
                        clean_truth = "".join(filter(str.isalpha, ground_truth))
                        
                        is_correct = 1 if clean_model == clean_truth else 0
                        writer.writerow(list(row_data) + [model_ans, is_correct])
                f.flush()
            except Exception as e:
                print(f"Error at index {i}: {e}")

# change input clue type manually
run_batch_solving("anagram.csv", "./solved/gemma_solved_anagrams.csv", "gemma", "anagram")
# run_batch_solving("anagram.csv", "./solved/llama_solved_anagrams.csv", "llama", "anagram")
# run_batch_solving("anagram.csv", "./solved/gpt_solved_anagrams.csv", "gpt", "anagram")