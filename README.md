# LLM Performance on Categorizing and Solving Different Types of Cryptic Crossword Clues

The /baselines and /decrypt directories are unused from the original repository. Most of our code is contained in /llm_prompting.
Rozner et al. (2021) is where Sadallah et al. (2025) sourced a majority of the data for their paper.

## Heuristic-based classification

To set the data up to run the heuristic-based classification script after cloning the repository, complete the following steps.  
1. Create a Python virtual environment and activate it (following command is for powershell):
```setup
python -m venv venv
.\venv\Scripts\Activate.ps1
```
2. Extract the disjoint.json.zip file into the ./data/ folder 
3. Install the dependencies from the Rozner et al. paper:
```setup
pip install -r requirements.txt
```
4. Run classifier.py in the /decrypt directory. Note that since we abandoned this approach early on, the categories do not match our LLM classification and solving tasks.

## LLM Classification and LLM Solving
The related scripts are all located in the /llm_prompting directory:
- load_csv.py converts the Rozner dataset from JSON to a CSV file
- llm_classifier.py takes in the original datset as a CSV and prompts the three models to identify the wordplay types of its clues
- split_classified.py separates the classified clues into one CSV file per wordplay type
- llm_solver.py prompts the models to solve the clues in the separated CSVs

The /classification and /solved subdirectories contain the output CSV files for their respective tasks. The /categories subdirectory contains the outputs of Gemma 4 classification, separated out by wordplay type.