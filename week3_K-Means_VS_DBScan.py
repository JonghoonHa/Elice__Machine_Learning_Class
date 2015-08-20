import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import numpy as np
import pandas as pd
import elice_utils

def main():
    np.random.seed(108)

    noisy_circles = pd.read_csv('noisy_circle.csv')  # 파일을 읽어들여 데이터 프레임으로 저장
    blobs = pd.read_csv('blobs.csv')
    
    return draw_graph([noisy_circles, blobs])


def run_kmeans(X, num_clusters):
    # 2차원으로 축소한 각 data points를 cluster화 한다.
    # 비슷한 성질을 가진 것 끼리 모아주는 역할인데,
    # 딱 정해진 답이 없으므로 여러번 시행한 평균 값을 결과로 갖는다.
    
    kmeans = sklearn.cluster.KMeans(num_clusters)
    kmeans.fit(X)

    return kmeans


def run_DBScan(X, eps):
    # DBScan 알고리즘은 K-Means에 비해 여러 장점이 있다.
    # 클러스터의 개수를 정하지 않아도 되고, 여러 모양의 클러스터를 찾을 수 있다.
    # eps는 같은 클러스터로 인식할 거리의 한계값을 말한다.
    #다시 말해, eps값보다 가까운 거리에 있는 값을 연결해 클러스터를 만드는 것이다.

    dbscan = sklearn.cluster.DBSCAN(eps)
    dbscan.fit(X)

    return dbscan


def draw_graph(datasets, n_clusters=2):
    
    # DB-scan 알고리즘에서 eps값이 변화함에 따라 같은 클러스터로 인식하는 범위가 달라지는 여부를 비교해볼 수 있다. 
    # 예상대로 eps값이 커질수록 멀리 떨어진 부분을 같은 클러스터로 인식한다.
    plot_num = 1
    alg_names = []
    eps_list = []

    for eps in np.arange(0.1, 1, step=0.1):
        alg_names.append('DBScan eps=%.1f' % eps)
        eps_list.append(eps)

    a = np.arange(len(alg_names)*2)+1
    indices = np.hstack(a.reshape(len(alg_names), 2).T)

    elice_utils.draw_init()

    for dataset in datasets:
        
        temp = []
        for idx in dataset.index :
            #print(str(dataset['1'][idx]) +', '+str(dataset['2'][idx]))
            temp.append([dataset['1'][idx], dataset['2'][idx]])
        
        X = np.array(temp)

        for alg_name, eps in zip(alg_names, eps_list):
            dbscan_result = run_DBScan(X, eps)

            elice_utils.draw_graph(X, dbscan_result, alg_name, plot_num, len(alg_names), indices)
            plot_num += 1

    print(elice_utils.show_graph())

    return dbscan_result


if __name__ == '__main__':
    main()  
