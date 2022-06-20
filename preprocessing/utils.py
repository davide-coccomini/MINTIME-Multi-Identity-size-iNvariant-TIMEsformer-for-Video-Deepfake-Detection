# Utility functions used in preprocessing steps

import glob
import os

# Returns all the file paths with a specific extension inside a requested root directory
def get_paths(rootdir, ext="png"):
    paths = []
    for path in glob.glob(f'{rootdir}/*/**/*.'+ext, recursive=True):
        paths.append(path)
    return paths






