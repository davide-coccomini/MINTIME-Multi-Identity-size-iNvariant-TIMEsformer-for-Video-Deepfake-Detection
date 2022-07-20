# Utility functions used in preprocessing steps

import glob
import os
from torchvision.transforms import Resize, ToPILImage, ToTensor
import networkx as nx

# Returns all the files paths with a specific extension inside a requested root directory
def get_paths(rootdir, ext="png"):
    paths = []
    for path in glob.glob(f'{rootdir}/*/**/*.'+ext, recursive=True):
        paths.append(path)
    return paths

# Cluster the images generating a graph of connected components
def _generate_connected_components(similarities, similarity_threshold=0.80):
    graph = nx.Graph()
    for i in range(len(similarities)):
        for j in range(len(similarities)):
            if i != j and similarities[i, j] > similarity_threshold:
                graph.add_edge(i, j)

    components_list = []
    for component in nx.connected_components(graph):
        components_list.append(list(component))
    graph.clear()
    graph = None

    return components_list

# Method used to preprocess the image before features extraction in clustering step
def preprocess_images(img, shape=[128, 128]):
    img = Resize(shape)(img)
    return img
