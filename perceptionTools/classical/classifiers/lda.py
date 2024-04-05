import numpy as np
import itertools

# Class for performing binary LDA
class LdaTwoclass:
    def __init__(self, dataset0: np.ndarray, dataset1: np.ndarray) -> None:
        # Initialize the LDA class, which includes training the model
        # Inputs:
        #   dataset0: 2D numpy array containing one sample from class 0 in each row
        #   dataset1: 2D numpy array containing one sample from class 1 in each row
        # Outputs:
        #   None

        # Compute class means
        self.m0 = np.mean(dataset0,axis=0)
        self.m1 = np.mean(dataset1,axis=0)
        # Compute class covariances
        c0 = np.cov(dataset0.T)
        c1 = np.cov(dataset1.T)
        # Fetch sample sizes
        n0 = dataset0.shape[0]
        n1 = dataset1.shape[0]
        # Compute pooled covariance
        pc = (n0*c0+n1*c1)/(n0+n1-2)
        # Compute projection vector originating from and orthogonal to separating hyperplane
        # This is the solution to an optimization problem that finds the dividing hyperplane that
        # minimizes the ratio of pooled within-class covariance to mean separation
        self.w = np.linalg.inv(pc)@(self.m1-self.m0)
    
    def classify(self, samples: np.ndarray) -> np.ndarray:
        # Given some samples to classify, produce classification score
        # Inputs:
        #   samples: 2D numpy array containing one sample to be classified in each row
        # Outputs:
        #   1D numpy array containing the class score, with positive numbers being class 1

        # Mean shift the samples so that a projection value of 0 is on the hyperplane
        p = samples - (self.m0+self.m1)/2
        # Compute the projection of the samples onto the vector defining the hyperplane
        return p@self.w

# Class for performing n-class LDA by one-v-all classification
# This will get ugly and inefficient for lots of classes
class LdaMultiClass:
    def __init__(self, datasets: list[np.ndarray]) -> None:
        # Initialize the LDA class, which includes training the model
        # Inputs:
        #   datasets: List of 2D numpy arrays. Each array represents a class with one sample per row
        # Outputs:
        #   None

        # Initialize list of classifiers
        self.lda2s = []
        for idx,not_class in enumerate(itertools.combinations(datasets,len(datasets)-1)):
            # Create out of class and in class datasets for binary LDA classifiers
            index_select = len(datasets)-idx-1
            dataset0 = np.vstack(not_class)
            dataset1 = datasets[index_select]
            # Create 1 vs all classifier for class idx
            self.lda2s.append(LdaTwoclass(dataset0,dataset1))
        self.lda2s.reverse()
    
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
        output_scores = np.zeros((samples.shape[0],len(self.lda2s)))

        # Iterate over classifiers
        for idx,classifier in enumerate(self.lda2s):
            output_scores[:,idx] = classifier.classify(samples)

        return output_scores, np.argmax(output_scores,axis=1)