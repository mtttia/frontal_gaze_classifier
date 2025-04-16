import kagglehub
import os

# Download latest version
dataset_path = kagglehub.dataset_download("kayvanshah/eye-dataset")

print("Path to dataset files:", dataset_path)