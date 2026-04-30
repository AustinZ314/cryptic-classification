import json
import pandas as pd

# load data from original dataset into a csv
def convert_json_to_csv(json_input_path, csv_output_path):
    with open(json_input_path, 'r') as f:
        raw_data = json.load(f)
        clue_list = raw_data.get("test", [])
    
    formatted_data = []
    for item in clue_list:
        formatted_data.append({
            "Index": item.get("idx"), 
            "Clue": item.get("clue"),
            "Solution": item.get("soln"),
            "Spaced_Solution": item.get("soln_with_spaces")
        })
    
    df = pd.DataFrame(formatted_data)
    df.to_csv(csv_output_path, index=False)
    print(f"Converted {len(df)} clues and saved to {csv_output_path}")

convert_json_to_csv("../rozner_data/naive_random.json", "full_dataset.csv")