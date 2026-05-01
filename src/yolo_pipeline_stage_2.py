from lib.dataset import merge_datasets, verify_dataset
from lib.baseline_validation import run_baseline_validation
from lib.training_tools import (
    profile_object_sizes,
    generate_yaml_config,
    generate_small_image_subset,
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
    # Run stage-2 YOLO training on the small-object dataset
    # ---------------------------------------------------------
    tracker = ExperimentTracker()

    dataset_path = "combined_dataset"
    yolo_model_name = "yolo26n-p2.yaml"
    baseline_weights = r"C:\Users\maxyi\Documents\ENGG2112\runs\detect\yolo26-hyperparams_baseline\weights\best.pt"
    run_name = "yolo26n-stage2-p2"
    epochs_to_run = 40
    fraction_size = 1
    image_size = 640
    batch_size = 4

    # Generate small-object subset and YAML config
    _, small_threshold = profile_object_sizes(dataset_path, sample_limit=12600, show_plot=False)
    small_txt_path, small_yaml_path, _ = generate_small_image_subset(
        dataset_path,
        small_threshold,
        output_txt_path="train_small.txt",
        yaml_path="c2a_small.yaml",
        sample_limit=12600,
        mosaic=1.0 # image_ext removed as it is now auto-detected
    )

    # Train using the small-object dataset and pretrained baseline weights
    results = train_yolo(
        yaml_path=small_yaml_path,
        yolo_model=yolo_model_name,
        weights=baseline_weights,
        img_size=image_size,
        epochs=epochs_to_run,
        batch_size=batch_size,
        fraction=fraction_size,
        project_dir=".",
        run_name=run_name,
        freeze=10, #to avoid catastrophic forgetting, freeze the backbone weight while getting the p2 head up to speed
        box=9, #more rigorous box for small objects since deviation will result in lower map
        lr0=0.005, #slower learning rate
        warmup_epochs=5, #increase warmup from 3 to 5 to avoid sudden jump
        scale=0.9 # scale objects increasing the size helps with training
    )

    final_map50 = getattr(results.box, "map50", None)
    final_map50_95 = getattr(results.box, "map", None)
    final_P = getattr(results.box, "mp", None)  # Mean Precision
    final_R = getattr(results.box, "mr", None)  # Mean Recall

    if final_map50 is None or final_map50_95 is None:
        raise RuntimeError("Unable to extract YOLO metrics from training results.")

    tracker.log(Experiment(
        name=run_name,
        model="yolov26n-p2",
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


# future work: 
# 1. filter all images with humans < threshold then use the stage 1 model to train with small humans only


if __name__ == '__main__':
    # This block safely executes multiprocessing on Windows
    main()