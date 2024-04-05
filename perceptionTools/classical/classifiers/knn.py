import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode

class KnnClassifier:
    def __init__(self, datasets: list[np.ndarray],metric: str='euclidean',**kwargs) -> None:
        # Initialize the KNN class, which includes training the model
        # Inputs:
        #   datasets: List of 2D numpy arrays. Each array represents a class with one sample per row
        #   metric: distance metric to use
        #     Both metric and kwargs to follow guidance give in
        #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # Outputs:
        #   None

        # Create array for class labels
        class_labels = []
        for idx in range(len(datasets)):
            class_labels.append(idx*np.ones((datasets[idx].shape[0],)))
        self.class_labels = np.hstack(class_labels)
        # Create stacked array for data
        self.datasets = np.vstack(datasets)
        # Store the desired distance metric exponent
        self.metric = metric
        self.kwargs = kwargs

    def classify(self,samples: np.ndarray, k: int) -> np.array:
        # Assigns a class label to each sample to be classified
        # Inputs:
        #   samples - 2D numpy array with one sample to be classified per row
        #   k - number of nearest neighbors to check against
        # Outputs:
        #   1D numpy array with one class label per row of the input

        # Error out if k greater than number of samples
        if k > self.datasets.shape[0]:
            raise ValueError("k must be <= number of points in the dataset")

        # Compute distance matrix between samples
        dist_mat = cdist(samples,self.datasets,self.metric,**self.kwargs)
        # Compute the argsort on this to get the nearest neighbors
        inds = np.argsort(dist_mat)
        # Fetch the class labels for the neighbors
        neighbor_class = self.class_labels[inds[:,:k]]
        # Compute the most voted for
        class_modes = mode(neighbor_class,axis=1,keepdims=False)
        # Extract mode and return
        return class_modes.mode