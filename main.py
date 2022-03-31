from numpy import float32, float64
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from KNN import KNeighborsClassifier
from GridSearch import GridSearch

filename="C:/Users/Lenovo/Documents/python/KNN/SDN_Intrusion.csv"

def knn(filedir):
    dataset = pd.read_csv(filedir)
    print('开始数据清洗')
    
    m = enumerate(dataset.iloc[:, -1].value_counts().keys())
    m = dict([(i[1], i[0]) for i in m])
    dataset.iloc[:, -1] = dataset.iloc[:, -1].map(m)
    
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.fillna(value=0)
    
    dataset.astype(float64)
    
    print('开始分割数据')
    x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:, 1:-1],
                                                        dataset.iloc[:, -1],
                                                        random_state=6)
    print('开始标准化')
    trainsfor = StandardScaler()
    x_train = trainsfor.fit_transform(x_train)
    x_test = trainsfor.transform(x_test)
    
    print('定义模型')
    estimator = KNeighborsClassifier()
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearch(estimator=estimator, param_grid=param_dict)

    print('开始训练')
    estimator.fit(x_train, y_train)

    print('开始预测')
    # y_predict = estimator.predict(x_test)
    #评估
    score = estimator.score(x_test, y_test)
    print("准确率\n", score)
    
    # print('best_k\n',estimator.best_k)


if __name__ == "__main__":
    knn(filename)