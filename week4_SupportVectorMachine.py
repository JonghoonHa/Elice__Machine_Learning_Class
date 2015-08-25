import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import sklearn.cross_validation as sc


def main():
    C = 1.0

    X, y = load_data()

    # 2
    X = sklearn.preprocessing.scale(X)

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    # Support Vector Machine의 kernel종류에 따른 분류 결과 비교
    svc_linear = run_linear_SVM(X_train, y_train, C)
    svc_poly2 = run_poly_SVM(X_train, y_train, 2, C)
    svc_poly3 = run_poly_SVM(X_train, y_train, 3, C)
    svc_rbf = run_rbf_SVM(X_train, y_train, C)

    model_names = ['Linear', 'Poly degree 2', 'Poly degree 3', 'RBF']
    for model_name, each_model in zip (model_names, [svc_linear, svc_poly2, svc_poly3, svc_rbf]):
        model_score = test_svm_models(X_test, y_test, each_model)
        print('%s score: %f' % (model_name, model_score))


def load_data():
    # 1
    
    dataset = pd.read_csv('blood_donation.csv')
    
    temp2 = dataset.pop('class')

    temp = []
    for idx in dataset.index :
        temp.append([dataset['Recency'][idx], dataset['Frequency'][idx], dataset['Monetary'][idx], dataset['Time'][idx]])
        
    X = np.array(temp)
    y = np.array(temp2)
    
    return X, y

def run_linear_SVM(X, y, C):
    # 2

    svc_linear = sklearn.svm.SVC(kernel='linear').fit(X, y)
    
    return svc_linear


def run_poly_SVM(X, y, degree, C):
    # 3

    svc_poly = sklearn.svm.SVC(kernel='poly', degree=degree).fit(X, y)

    return svc_poly


def run_rbf_SVM(X, y, C, gamma=0.7):
    # 4

    svc_rbf = sklearn.svm.SVC(kernel='poly', gamma=gamma, C=C).fit(X, y)

    return svc_rbf


def test_svm_models(X_test, y_test, each_model):
    # 6
    
    print(each_model)
    
    score_value = sc.cross_val_score(X_test, y_test)

    return score_value


if __name__ == "__main__":
    main()
