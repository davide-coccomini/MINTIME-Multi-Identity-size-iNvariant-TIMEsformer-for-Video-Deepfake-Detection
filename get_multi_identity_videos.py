import os
import pandas as pd

DATA_CSV = "../../datasets/dfdc_test_preview/test_videos_preview_labels.csv"
DATA_PATH = "../../datasets/dfdc_test_preview/faces"


col_names = ["video", "label"]
df_test = pd.read_csv(DATA_CSV, sep=' ', names=col_names)

indexes_to_drop = []
for index, row in df_test.iterrows():
    folders = os.listdir(os.path.join(DATA_PATH, row['video']))
    if len(folders) < 2:
        indexes_to_drop.append(index)
    else:
        counter = 0
        for folder in folders:
            if os.path.isdir(os.path.join(DATA_PATH, row['video'], folder)):
                counter += 1
        if counter < 2:
            indexes_to_drop.append(index)

df_test.drop(df_test.index[indexes_to_drop], inplace=True)


df_test.to_csv("../../datasets/dfdc_test_preview/multi_identity_videos.csv")

print(len(df_test))


