import kagglehub

# Download latest version
path = kagglehub.dataset_download("rgbnihal/c2a-dataset")

print("Path to dataset files:", path)