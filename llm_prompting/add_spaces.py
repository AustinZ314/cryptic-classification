import pandas as pd
import json

def add_spaced_solutions(json_source, classified_csv, output_csv):
    with open(json_source, 'r') as f:
        raw_data = json.load(f)
        clue_list = raw_data.get("test", [])
        
    spaced_map = {item['idx']: item.get('soln_with_spaces', '') for item in clue_list}
    
    df_results = pd.read_csv(classified_csv)
    df_results['Spaced_Solution'] = df_results['Index'].map(spaced_map)
    
    df_results.to_csv(output_csv, index=False)
    print(f"Solutions with spaces added. New file saved as: {output_csv}")

add_spaced_solutions("../rozner_data/naive_random.json", "classified_dataset.csv", "classified_with_spaces.csv")