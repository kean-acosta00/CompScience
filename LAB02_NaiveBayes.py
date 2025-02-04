class NaiveBayes:
    def __init__(self):
        self.category_probs = {}
        self.feature_probs = {}
        self.categories = set()

    def train(self, features, labels):
        self.categories = set(labels)
        total_samples = len(labels)

        for category in self.categories:
            self.category_probs[category] = labels.count(category) / total_samples

        self.feature_probs = {category: {} for category in self.categories}

        for category in self.categories:
            category_samples = [features[i] for i in range(len(labels)) if labels[i] == category]
            num_features = len(features[0])

            for j in range(num_features):
                feature_values = [sample[j] for sample in category_samples]
                unique_values = set(feature_values)

                self.feature_probs[category][j] = {}

                for value in unique_values:
                    self.feature_probs[category][j][value] = (feature_values.count(value) + 1) / (len(category_samples) + len(unique_values))

    def classify(self, new_features):
        if not isinstance(new_features[0], list):
            new_features = [new_features]

        predictions = []
        for sample in new_features:
            category_scores = {}

            for category in self.categories:
                category_scores[category] = self.category_probs[category]

                for j in range(len(sample)):
                    value = sample[j]
                    if value in self.feature_probs[category][j]:
                        category_scores[category] *= self.feature_probs[category][j][value]
                    else:
                        category_scores[category] *= 1e-6

            best_category = max(category_scores, key=category_scores.get)
            predictions.append(best_category)

        return predictions

# DATASET
data_samples = [
    [1, 'Low'], [1, 'Medium'], [1, 'Medium'], [1, 'Low'], [1, 'Low'],
    [2, 'Low'], [2, 'Medium'], [2, 'Medium'], [2, 'High'], [2, 'High'],
    [3, 'High'], [3, 'Medium'], [3, 'Medium'], [3, 'High'], [3, 'High']]
class_labels = ['Reject', 'Reject', 'Accept', 'Accept', 'Reject',
                'Reject', 'Reject', 'Accept', 'Accept', 'Accept',
                'Accept', 'Accept', 'Accept', 'Accept', 'Reject']

classifier = NaiveBayes()
classifier.train(data_samples, class_labels)

# PREDICTION EXAMPLE
test_samples = [[2, 'Low']]
predictions = classifier.classify(test_samples)
print("Predictions:", predictions)