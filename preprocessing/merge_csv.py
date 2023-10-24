import pandas as pd


df1 = pd.read_csv("../../../datasets/dfdc_test_preview/test_videos_preview_labels.csv", sep=' ', names=["name", "label"])
df2 = pd.read_csv("preview.csv", sep=' ', usecols=["name", "label"])


df3 = df1.merge(df2, on=["name"])

df3 = df3.drop(["label_x"], axis=1)
df3.to_csv("common.csv", index=False)