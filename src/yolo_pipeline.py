from lib.dataset import merge_datasets, verify_dataset
from lib.baseline_validation import run_baseline_validation
from lib.training_tools import ( 
    profile_object_sizes, 
    generate_yaml_config, 
    train_yolo,
    ExperimentTracker, plot_results, Experiment
)

def main():
    # --- 1. Download and merge datasets ---
    print("Step 1: Downloading and merging datasets...")
    merge_datasets()
    
    # --- 2. Run baseline validation ---
    # print("\nStep 2: Running baseline validation...")
    # run_baseline_validation()
    
    # ---------------------------------------------------------
    # Run a real YOLO training pipeline and log the results
    # ---------------------------------------------------------
    tracker = ExperimentTracker()

    dataset_path = "combined_dataset"
    yaml_config_path = "combined_local.yaml"
    run_name = "yolo26-test"
    yolo_model_name = "yolo26n.pt"
    epochs_to_run = 1
    fraction_size = 1 #0 to 1
    image_size = 800

    # Generate or refresh the YAML config from the current dataset
    generate_yaml_config(dataset_path, yaml_path=yaml_config_path)

    # Train using the real pipeline and capture metrics
    results = train_yolo(
        yaml_path=yaml_config_path,
        yolo_model=yolo_model_name,
        img_size= image_size,
        epochs=epochs_to_run,
        batch_size=8,
        fraction=1,
        project_dir=".",
        run_name=run_name
    )

    final_map50 = getattr(results.box, "map50", None)
    final_map50_95 = getattr(results.box, "map", None)
    final_P = getattr(results.box, "mp", None)  # Mean Precision
    final_R = getattr(results.box, "mr", None)  # Mean Recall

    if final_map50 is None or final_map50_95 is None:
        raise RuntimeError("Unable to extract YOLO metrics from training results.")

    tracker.log(Experiment(
        name=run_name,
        model="yolov26n",
        map50=final_map50,
        map50_95=final_map50_95,
        epochs=epochs_to_run,
        fraction=fraction_size,
        img_size=image_size,
        P=final_P,
        R=final_R
    ))

    print(f"Logged {len(tracker)} real YOLO experiments.")
    print(f"Best by mAP@50    : {tracker.best_by('map50')}")
    print(f"Best by mAP@50-95 : {tracker.best_by('map50_95')}")

    plot_results(tracker)


# future work: filter all images with humans < threshold then use the stage 1 model to train with small humans only


if __name__ == '__main__':
    # This block safely executes multiprocessing on Windows
    main()