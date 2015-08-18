import numpy as np
import sklearn
import sklearn.metrics
import sklearn.cluster
import sklearn.datasets
import sklearn.decomposition
import sklearn.preprocessing
import elice_utils

def main():
    # 1
    digits = sklearn.datasets.load_digits()
    data = sklearn.preprocessing.scale(digits.data)

    # 2
    # Try looking at different digits by changing index from 0 to 1796.
    #print(elice_utils.display_digits(digits, 113))

    # 4
    benchmark(data, digits.target, 1, 64)

def benchmark(data, ground_truth, components_min, components_max):
    np.random.seed(0)

    X = []
    Y = []
    for num_components in range(components_min, components_max):
        X.append(num_components)
        pca_array = run_PCA(data, num_components) # 1-64 차원의 데이터로 압축
        estimated_classes = run_kmeans(pca_array, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # 4
        score = sklearn.metrics.homogeneity_score(ground_truth, estimated_classes)

        Y.append(score)

    print(elice_utils.benchmark_plot(X, Y))
    return Y

def run_PCA(df, num_components):
   # Run PCA
    # PCA 알고리즘은 고차원의 독립변수를 가진 데이터를 낮은 차원으로 압축한다.
    # 분산값이 가장 큰 것을 우선하여 낮은 차원으로 단순화한다.
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(df)
    pca_array = pca.transform(df)

    return pca_array

def run_kmeans(champ_pca_array, num_clusters, initial_centroid_indices):
    # Implement K-Means algorithm using sklearn.cluster.KMeans.
    # 2차원으로 축소한 각 data points를 cluster화 한다.
    # 비슷한 성질을 가진 것 끼리 모아주는 역할인데,
    # 딱 정해진 답이 없으므로 여러번 시행한 평균 값을 결과로 갖는다.
    
    # initial_centroids (초기 중점이 될 위치) 선정
    X = np.array(champ_pca_array)
    initial_centroids = X[initial_centroid_indices]
    
    # k-Means algorithm 작동
    classifier = sklearn.cluster.KMeans(num_clusters, initial_centroids ,1)
    classifier.fit(champ_pca_array)

    return classifier.labels_

if __name__ == "__main__":
    main()
