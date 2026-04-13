# ENGG2112
Codebase for ENGG2112 group project

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/max-yip/ENGG2112.git
   cd ENGG2112
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

To run the complete pipeline, execute the scripts in the following order:

1. Merge the datasets:
   ```bash
   python src/merge_dataset.py
   ```

2. Run baseline validation:
   ```bash
   python src/baseline_validation.py
   ```

3. Run the main training script which contains functions from the training_pipeline.py:
   ```bash
   python src/main.py
   ```
