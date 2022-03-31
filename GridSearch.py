from sklearn.model_selection import train_test_split
import numpy as np

class GridSearch:
    def __init__(self,estimator, param_grid):
        self.param=param_grid["n_neighbors"]
        self.model=estimator
        self.best_k=3

    def fit(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=6)
        self.model.fit(x_train,y_train)
        dic=self.__score(x_test,y_test)
        print('k : accuracy\n',dic)
        self.best_k=max(dic,key=dic.get)

    def __predict(self,x):
        dic={}
        for i in self.param:
            dic[i]=self.model.predict(x,i)
        return dic
    
    def __score(self,x,y):
        dic=self.__predict(x)
        ans={}
        for i,x in dic.items():
            ans[i]=sum(x == y)/len(x)
        return ans
    def predict(self,x):
        return self.model.predict(x,self.best_k)
    def score(self,x,y):
        y = np.array(y)
        x = self.predict(x)
        return sum(x == y) / len(x)