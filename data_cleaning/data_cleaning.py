import pandas as pd
import os
import zipfile

zip_path = "learning_traces.13m.csv.zip"
data_path = "learning_traces.13m.csv"
output_dir = "cleaned_data"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(data_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print(f"Extracted {data_path} from {zip_path}")
else:
    print(f"{data_path} already exists.")

use_cols = ['p_recall', 'delta', 'learning_language', 'history_seen', 'history_correct']
chunks = pd.read_csv(data_path, usecols=use_cols, chunksize=500000)
df = pd.concat(chunks, ignore_index=True)

df = df.dropna(subset=['p_recall', 'delta', 'learning_language'])
df['learning_language'] = df['learning_language'].astype(str).str.strip()

df.to_csv(os.path.join(output_dir, "duolingo_all_languages.csv"), index=False)
print("Saved cleaned data with all languages.")

languages = df['learning_language'].unique()
print(f"Detected {len(languages)} languages: {languages}")

for lang in languages:
    subset = df[df['learning_language'] == lang]
    file_name = f"duolingo_{lang}.csv"
    subset.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f"Saved cleaned data for language: {lang} to {file_name}")

print("Data cleaning and splitting completed.")
print(f"Total cleaned rows: {len(df):,}")