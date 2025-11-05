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

lang_counts = df['learning_language'].value_counts(normalize=True)
major_langs = lang_counts[lang_counts >=0.10].index.tolist()
print(f"Major languages (>=10%): {major_langs}")

df.to_csv(os.path.join(output_dir, "duolingo_all_languages.csv"), index=False)
print("Saved cleaned data with all languages.")

for lang in major_langs:
    subset = df[df['learning_language'] == lang]
    subset.to_csv(os.path.join(output_dir, f"duolingo_{lang}.csv"), index=False)
    print(f"Saved duolingo_{lang}.csv")

other_df = df[~df['learning_language'].isin(major_langs)]
if not other_df.empty:
    other_df.to_csv(os.path.join(output_dir, "duolingo_other_languages.csv"), index=False)
    print("Saved duolingo_other_languages.csv")

print("Data cleaning and segmentation completed.")
print(f"Total cleaned rows: {len(df):,}")
print(f"Languages >=10%: {len(major_langs)} | Other languages: {len(lang_counts) - len(major_langs)}")