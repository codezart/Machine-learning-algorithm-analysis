import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = int(k)
        self._fit_data = []
    
    def fit(self, x, y):

        assert len(x) == len(y)
        self._fit_data = [(Point(coordinates), label) for coordinates, label in zip(x, y)]
    
    def predict(self, x):
        predicts = []
        for coordinates in x:
            predict_point = Point(coordinates)

            # euclidean distance from predict_point to all in self._fit_data
            distances = []
            for data_point, data_label in self._fit_data:
                distances.append((predict_point.distance(data_point), data_label))

            # k points with less distances
            distances = sorted(distances, key=itemgetter(0))[:self.k]
            # label of k points with less distances
            predicts.append(list(max(distances, key=itemgetter(1))[1]))

        return predicts

class Point:

    def __init__(self, axis):
        self.axis = np.array(axis)
    
    def distance(self, other):
        # if not isinstance(other, Point):
        #     other = Point(other)
        # Euclidean distance
        return sum((self - other) ** 2) ** 0.5

if __name__=="__main__":
    iris = load_iris()
    print(iris.head())

