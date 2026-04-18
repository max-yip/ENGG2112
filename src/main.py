from merge_dataset import merge_datasets
from baseline_validation import run_baseline_validation
from training_pipeline import (
    verify_dataset, 
    profile_object_sizes, 
    generate_yaml_config, 
    train_yolo
)

def main():
    # --- 1. Download and merge datasets ---
    print("Step 1: Downloading and merging datasets...")
    merge_datasets()
    
    # --- 2. Run baseline validation ---
    print("\nStep 2: Running baseline validation...")
    run_baseline_validation()
    
    # --- Optional: Continue with training pipeline ---
    print("\nStep 3: Starting training pipeline...")
    dataset_path = "combined_dataset"
    yaml_config_path = "combined_local.yaml"
    
    # Step 1: Make sure our data paths are correct
    verify_dataset(dataset_path)
    
    # Step 2: Data profiling (you can set show_plot=False to run headlessly)
    profile_object_sizes(dataset_path, sample_limit=6129, show_plot=True)
    
    # Step 3: Auto-generate the dataset YAML
    generate_yaml_config(dataset_path, yaml_path=yaml_config_path)
    
    # Step 4: Kick off GPU training
    train_yolo(
        yaml_path=yaml_config_path, 
        epochs=10, 
        batch_size=8, 
        fraction=0.1,  # 10% of data for rapid prototyping
        project_dir=".", 
        run_name="stage1_all_data"
    )

if __name__ == '__main__':
    # This block safely executes multiprocessing on Windows
    main()