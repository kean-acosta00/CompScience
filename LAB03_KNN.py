class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.features = []
        self.labels = []

    # FOR STORING THE TRAINING DATA AND LABELS
    def train(self, features, labels):
        if not features or not labels or len(features) != len(labels):
            raise ValueError("Training data and labels must be non-empty and of equal length.")
        self.features = features
        self.labels = labels

    # PREDICTING THE LABELS FOR A LIST OF POINTS
    def predict(self, test_points):
        return [self._predict_single(point) for point in test_points]

    # FINDING K-NN AND RETURNING THE COMMON LABEL
    def _predict_single(self, test_point):
        distances = [(self._euclidean_distance(test_point, train_point), label)
                     for train_point, label in zip(self.features, self.labels)]
        
        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:self.k]]
        
        return self._majority_vote(nearest_labels)

    # EUCLIDEAN CALCULATION
    def _euclidean_distance(self, point1, point2):
        return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

    def _majority_vote(self, labels):
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)

# DATASET
if __name__ == "__main__":
    train_data = [[3, 7], [2, 9], [6, 1], [8, 5], [1, 4], [7, 2], [5, 8], [4, 6]]
    train_labels = ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B']
    
    # POINTS TO PREDICT
    test_data = [[4, 6], [7, 3]]

    knn = KNearestNeighbors(k=3)
    knn.train(train_data, train_labels)
    predictions = knn.predict(test_data)
    
    print("Predictions:", predictions)