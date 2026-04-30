import pandas as pd
import os

# split classified clues into separate csv files by wordplay type
def split_by_category(input_csv, output_dir="categories"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(input_csv)
    
    pred_col = [col for col in df.columns if "_predicted" in col][0]
    
    categories = df[pred_col].unique()
    
    for category in categories:
        filename = f"{str(category).replace(' ', '_').lower()}.csv"
        category_df = df[df[pred_col] == category]
        
        output_path = os.path.join(output_dir, filename)
        category_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(category_df)} clues.")

split_by_category("./classification/classified_dataset.csv")