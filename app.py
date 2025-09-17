import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np
import zipfile

# ------------------- Local Dataset Setup -------------------
DATA_ZIP_PATH = "fake.csv.zip"   # your uploaded file
DATA_FOLDER = "data"
DATA_CSV_PATH = os.path.join(DATA_FOLDER, "fake.csv")

# Create data folder if not exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Unzip if not already extracted
if not os.path.exists(DATA_CSV_PATH):
    print("Unzipping local dataset...")
    with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_FOLDER)
    print("✅ Dataset ready!")

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description='Fake News image downloader')
parser.add_argument(
    '--type',
    type=str,
    default=DATA_CSV_PATH,   # default points to extracted file
    help='Path to train, validate, or test file (default: data/fake.csv)'
)
args = parser.parse_args()

# ------------------- Load Dataset -------------------
print(f"Loading dataset: {args.type}")
df = pd.read_csv(args.type)
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
print("🎉 All done!")
