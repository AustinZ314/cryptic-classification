# LLM Performance on Categorizing and Solving Different Types of Cryptic Crossword Clues

## Setup

To set the data up to run the classification script after cloning the repository, complete the following steps.  
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