import numpy as np

# Class for performing n-class classification via malaanobis distance
# by one-v-all classification
class Mahalanobis:
    def __init__(self, datasets: list[np.ndarray]) -> None:
        # Initialize the Mahalanobis class, which includes training the model
        # Inputs:
        #   datasets: List of 2D numpy arrays. Each array represents a class with one sample per row
        # Outputs:
        #   None

        # Initialize list of classifiers
        self.mean = []
        self.cov = []
        for class_data in datasets:
            # Compute class mean and covariance matrix
            self.mean.append(np.mean(class_data,axis=0))
            self.cov.append(np.cov(class_data.T))
    
    def classify(self, samples: np.ndarray) -> np.ndarray:
        # Given some samples to classify, produce classification scores for each 1 v all
        # classifier and predicted class
        # Inputs:
        #   samples: 2D numpy array containing one sample to be classified in each row
        # Outputs:
        #   2D numpy array with each column containing the scores for one sample across all
        #     1 v all classifiers
        #   1D numpy array with predicted class for each sample

        # Initialize output for scores
        output_scores = np.zeros((samples.shape[0],len(self.mean)))

        # Iterate over classifiers
        for idx,params in enumerate(zip(self.mean,self.cov)):
            class_mean,class_cov = params
            output_scores[:,idx] = np.sqrt(np.sum(np.power(np.linalg.inv(class_cov)@(samples-class_mean).T,2),axis=0))

        return output_scores, np.argmin(output_scores,axis=1)