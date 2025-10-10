import os
import time
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import requests  # Import the requests library to handle HTTP errors

# Define the dataset repo and local directory
dataset_repo = "bghira/pseudo-camera-10k"
dataset_dir = "datasets"
dreambooth_subject_dir = os.path.join(dataset_dir, "dreambooth-subject")

# Function to download the dataset with retry logic
def download_dataset_with_retry(repo_id, retries=5, delay=60):
    # Try to download the dataset, retrying in case of rate limit
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries}...")
            dataset = load_dataset(repo_id)
            print("Dataset successfully downloaded.")
            return dataset  # Return the dataset if successful
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate-limited error
                print("Rate limit hit! Waiting before retrying...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff: increase delay for next retry
                continue
            else:
                raise e  # Raise other HTTP errors
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(delay)  # Wait before retrying
            delay *= 2  # Exponential backoff
            continue
    print("Failed to download dataset after several attempts.")
    return None

# Create the directories if they don't exist
os.makedirs(dreambooth_subject_dir, exist_ok=True)

# Call the function to download the dataset with retry logic
dataset = download_dataset_with_retry(dataset_repo)

# If the dataset is downloaded, proceed with your logic
if dataset:
    print(f"Dataset downloaded and ready at {dataset_dir}.")
    print(f"Place your images into the {dreambooth_subject_dir} directory now.")
else:
    print("Dataset download failed.")
