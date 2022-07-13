# Utility functions used in preprocessing steps

import glob
import os
from torchvision.transforms import Resize, ToPILImage, ToTensor
import networkx as nx

# Returns all the file paths with a specific extension inside a requested root directory
def get_paths(rootdir, ext="png"):
    paths = []
    for path in glob.glob(f'{rootdir}/*/**/*.'+ext, recursive=True):
        paths.append(path)
    return paths


def _generate_connected_components(similarities, similarity_threshold=0.80):
    # create graph
    graph = nx.Graph()
    for i in range(len(similarities)):
        for j in range(len(similarities)):
            if i != j and similarities[i, j] > similarity_threshold:
                graph.add_edge(i, j)

    components_list = []
    # for component in nx.strongly_connected_components(graph):
    for component in nx.connected_components(graph):
        components_list.append(list(component))
    graph.clear()
    graph = None

    return components_list


def preprocess_images(img, shape=[128, 128]):
    """
    Preprocess the images. Transforms them to PIL and resizes them
    Parameters
    ----------
    img : str the input image in numpy array format (h,w,c) shape : list the
        resulting shape (default is [128,128])
    """

    #img = ToPILImage("RGB")(img)
    img = Resize(shape)(img)
    return img
