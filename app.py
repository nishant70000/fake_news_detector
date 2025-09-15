import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np
import gdown
import zipfile

# ------------------- Google Drive Setup -------------------
FILE_ID = "1a9VsnDJlwTd63JqDVLeycZNbO_SmFJFB"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
DATA_ZIP_PATH = "data/train.tsv.zip"
DATA_TSV_PATH = "data/train.tsv"

# Create data folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Download and unzip if not exists
if not os.path.exists(DATA_TSV_PATH):
    print("Downloading dataset from Google Drive...")
    gdown.download(URL, DATA_ZIP_PATH, quiet=False, fuzzy=True)

    print("Unzipping dataset...")
    with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("Dataset ready!")

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')
parser.add_argument(
    '--type',
    type=str,
    default=DATA_TSV_PATH,   # default points to extracted file
    help='Path to train, validate, or test file (default: data/train.tsv)'
)
args = parser.parse_args()

# ------------------- Load Dataset -------------------
print(f"Loading dataset: {args.type}")
df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# ------------------- Image Downloader -------------------
pbar = tqdm(total=len(df))

if not os.path.exists("images"):
    os.makedirs("images")

for index, row in df.iterrows():
    if str(row.get("hasImage", "")) == "True" and row.get("image_url", "") not in ["", "nan"]:
        try:
            image_url = row["image_url"]
            urllib.request.urlretrieve(image_url, f"images/{row['id']}.jpg")
        except Exception as e:
            print(f"Failed to download {row.get('id', 'unknown')}: {e}")
    pbar.update(1)

pbar.close()
print("✅ All done!")
