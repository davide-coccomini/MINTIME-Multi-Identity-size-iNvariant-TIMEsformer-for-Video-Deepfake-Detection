import os
import csv
import glob

DATA_PATH = "../../datasets/ForgeryNet/faces"
paths = glob.glob(f'{DATA_PATH}/*/**/*.png', recursive=True)
print(len(paths))

with open('../csv/faces_files_structure.csv', 'w+') as f:
    for path in paths:
        f.write(path + "\n")
