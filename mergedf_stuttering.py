import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm

# Load dataset
df = pd.read_csv('fluencybank_labels.csv')

# Adding Name Column
df['Name'] = df[df.columns[0:3]].apply(
    lambda x: '_'.join(x.dropna().astype(str)),
    axis=1
)

# Remove empty audio files
CLIPS_DIR = "clips/"
ignore_list = []

for filename in os.listdir(CLIPS_DIR):
    file_path = os.path.join(CLIPS_DIR, filename)
    if 'FluencyBank' in filename:
        if os.stat(file_path).st_size == 44:
            ignore_list.append(filename)
            filename = filename[:-4]
            df = df[df.Name != filename]

print(len(ignore_list))

# MFCC Feature Extraction
features = {}

for filename in tqdm(os.listdir(CLIPS_DIR)):
    filename = filename[:-4]
    if 'FluencyBank' in filename and filename + '.wav' not in ignore_list:
        audio, sample_rate = librosa.load(os.path.join(CLIPS_DIR, filename + '.wav'), res_type='kaiser_fast', sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
        features[filename] = mfccs

print(len(features))

# Create dataset from features
df_features = pd.DataFrame.from_dict(features)
df_features = df_features.transpose()
df_features = df_features.reset_index()
df_features.rename(columns={'index': 'Name'}, inplace=True)

# Merge with original dataframe
df_final = pd.merge(df, df_features, how='inner', on='Name')

# Save the final dataframe
df_final.to_csv('final_dataset.csv', index=False)


