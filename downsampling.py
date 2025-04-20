import pandas as pd
import os
import shutil


df = pd.read_csv("train.csv")
df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(x))
df_0 = df[df['label'] == 0].sample(n=5000, random_state=42) # downsampling 5k samples each
df_1 = df[df['label'] == 1].sample(n=5000, random_state=42)

sampled_df = pd.concat([df_0, df_1]).sample(frac=1.0, random_state=42).reset_index(drop=True)


new_img_dir = "train_data_sampled"
os.makedirs(new_img_dir, exist_ok=True)

original_img_dir = "train_data"

for fname in sampled_df["file_name"]:
    src = os.path.join(original_img_dir, fname)
    dst = os.path.join(new_img_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"error: {src}")

sampled_df.to_csv("train_sampled.csv", index=False)

