import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt 

CSV_PATH_TRAIN = "../../../datasets/ForgeryNet/faces/train_and_val.csv"

CSV_PATH_TEST = "../../../datasets/ForgeryNet/faces/test.csv"
DATA_PATH = "../../../datasets/ForgeryNet/faces/"

col_names = ["video", "label", "8_cls"]

df_train = pd.read_csv(CSV_PATH_TRAIN, sep=' ', names=col_names)

df_test = pd.read_csv(CSV_PATH_TEST, sep=' ', names=col_names)
counters_train_test = []
for df in [df_train, df_test]:
    indexes_to_drop = []
    for index, row in df.iterrows():
        video_path = os.path.join(DATA_PATH, row["video"])
        if not os.path.exists(video_path) or len(os.listdir(video_path)) == 0:
            indexes_to_drop.append(index)
    df.drop(df.index[indexes_to_drop], inplace=True)


    identities_numbers = []
    for row in df.iterrows():
        video_path = os.path.join(DATA_PATH, row[1]["video"])
        identities = len(os.listdir(video_path))
        identities_numbers.append(identities)

    counters = Counter(identities_numbers)
    counters_train_test.append(counters)


total_identities_train = sum(counters_train_test[0].values())
total_identities_test = sum(counters_train_test[1].values())

collapsed_train_count = sum(count for num_identities, count in counters_train_test[0].items() if num_identities >= 4)
collapsed_test_count = sum(count for num_identities, count in counters_train_test[1].items() if num_identities >= 4)

counters_train_test[0][4] = collapsed_train_count
counters_train_test[1][4] = collapsed_test_count

data = {
    'Number of identities': list(range(1, 4)) + ['4+'],
    'Train': [counters_train_test[0][i] for i in range(1, 4)] + [counters_train_test[0][4]],
    'Test': [counters_train_test[1][i] for i in range(1, 4)] + [counters_train_test[1][4]]
}

df_plot = pd.DataFrame(data)

df_plot['Number of identities'] = df_plot['Number of identities'].apply(lambda x: '4+' if x == 4 else str(x))

plt.figure(figsize=(8, 6))
bar_width = 0.35
opacity = 0.8

plt.bar(df_plot.index, df_plot['Train'], bar_width, alpha=opacity, color='b', label='Train')
plt.bar([x + bar_width for x in df_plot.index], df_plot['Test'], bar_width, alpha=opacity, color='g', label='Test')

plt.xlabel('Number of identities')
plt.ylabel('Number of videos')
plt.title('Number of videos by number of identities (Train and Test)')
plt.xticks([r + bar_width/2 for r in range(len(df_plot))], df_plot['Number of identities'])  
plt.legend()

output_path = "../outputs/plots/forgerynet_multiidentity_videos.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)

print(counters_train_test)