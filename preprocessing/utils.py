#################################################################################################################
####################              Utility functions used in many parts of the repo           ####################
#################################################################################################################

import glob
import os

def get_paths(rootdir, ext="png"):
    paths = []
    for path in glob.glob(f'{rootdir}/*/**/*.'+ext, recursive=True):
        paths.append(path)
    return paths






