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

To run the available pipeline scripts, use the actual files in the repository:

1. Merge the datasets:
   ```bash
   python src/lib/dataset.py
   ```

2. Run baseline validation on the merged dataset:
   ```bash
   python src/lib/baseline_validation.py
   ```

3. Run the YOLO training pipeline:
   ```bash
   python src/yolo_pipeline.py
   ```

4. Optional alternative pipeline entry points:
   ```bash
   python src/faster_rcnn_pipeline.py
   python src/retinanet_pipeline.py
   python src/yolo_pipeline_stage_2.py
   ```

These scripts use the local dataset config files such as `combined_local.yaml` and `dataset_local.yaml`, and the provided YOLO weights like `yolov8n.pt`, `yolo26s.pt`, and `yolo26n.pt`. If you want to rebuild the merged dataset, delete `combined_dataset/` before rerunning `src/lib/dataset.py`.