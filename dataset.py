import os
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class DatasetProcessor:
    def __init__(self, n_samples=400, noise=0.2, random_state=42):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.dataset = None
        self.splits = {}

    def create_dataset(self):
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        self.dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        self.dataset['Target'] = y
        return self.dataset

    def split_dataset(self):
        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test)
        }
        return self.splits
