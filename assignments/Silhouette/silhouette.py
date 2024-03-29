import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

def load_data(file_name):
    return pd.read_csv(file_name)

def determine_clusters(data):
    return data['Cluster'].unique()

def calculate_cohesion(data, point, cluster):
    cluster_data = data[data['Cluster'] == cluster].drop(['Cluster', 'Instance'], axis=1)
    if cluster_data.shape[0] > 1:  # Ensure there is more than one point to avoid division by zero
        distances = pairwise_distances([point], cluster_data, metric='euclidean')
        return np.mean(distances)
    else:
        return 0

def calculate_separation(data, point, cluster):
    min_distance = np.inf
    for other_cluster in determine_clusters(data):
        if other_cluster == cluster:
            continue
        other_cluster_data = data[data['Cluster'] == other_cluster].drop(['Cluster', 'Instance'], axis=1)
        distances = pairwise_distances([point], other_cluster_data, metric='euclidean')
        average_distance = np.mean(distances)
        if average_distance < min_distance:
            min_distance = average_distance
    return min_distance

def calculate_silhouette(data):
    clusters = determine_clusters(data)
    silhouette_scores = []

    for index, row in data.iterrows():
        cluster = row['Cluster']
        point = row.drop(['Cluster', 'Instance']).to_numpy()
        ai = calculate_cohesion(data, point, cluster)
        bi = calculate_separation(data, point, cluster)
        if bi == 0:  # Avoid division by zero
            si = 0
        else:
            si = abs(1 - (ai / bi))
        silhouette_scores.append(si)

    return np.mean(silhouette_scores)

if __name__ == "__main__":
    k_values = [2, 3, 5, 7]
    for k in k_values:
        file_name = f'C:\\Users\\vpark\\Vee\\DSCI\\DSCI-633\\assignments\\data\\k{k}.csv'
        data = load_data(file_name)
        silhouette_score = calculate_silhouette(data)
        print(f'Avg silhouette coefficient for k of {k} = {silhouette_score:.3f}')
