class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    # RANDOM SELECTION
    def fit(self, data):
        self.centroids = [data[i] for i in range(self.k)]
        
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(data)
            new_centroids = self.compute_new_centroids(clusters)

            if self.has_converged(new_centroids):
                break

            self.centroids = new_centroids

    def assign_clusters(self, data):
        clusters = {i: [] for i in range(self.k)}

        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(point)

        return clusters

    def compute_new_centroids(self, clusters):
        new_centroids = []
        for i in range(self.k):
            if clusters[i]: 
                avg_x = sum(p[0] for p in clusters[i]) / len(clusters[i])
                avg_y = sum(p[1] for p in clusters[i]) / len(clusters[i])
                new_centroids.append([avg_x, avg_y])
            else:
                new_centroids.append(self.centroids[i])  # KEEPS PREVIOUS CENTROID IF NO POINTS ARE ASSIGNED/UPDATED
        return new_centroids

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 
    def has_converged(self, new_centroids):
        return all(
            abs(self.centroids[i][j] - new_centroids[i][j]) < 1e-5
            for i in range(self.k) for j in range(len(self.centroids[i]))
        )

    def predict(self, data):
        return self.assign_clusters(data)

# DATASET
if __name__ == "__main__":
    points = [[2, 3], [3, 4], [5, 8], [8, 7], [9, 6], [7, 5], [1, 2], [6, 9]]
    model = KMeans(k=3)
    model.fit(points)
    grouped_points = model.predict(points)

    print("Final Centroids:", model.centroids)
    for group_id, pts in grouped_points.items():
        print(f"Cluster {group_id + 1}: {pts}")