from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FailurePrediction:

    def __init__(self):

        self.model = RandomForestClassifier()

        self.X = []
        self.y = []

    def add_example(self, features, label):

        x = [
            features['loss_slope'],
            features['avg_grad_norm'],
            features['grad_variance'],
        ]

        self.X.append(x)

        self.y.append(label)

    def train(self):

        self.model.fit(np.array(self.X), np.array(self.y))
