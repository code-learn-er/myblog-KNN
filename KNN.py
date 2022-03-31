import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from collections import deque, Counter


class mergesort:
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis
        self.mgsort(0, len(a) - 1)

    def merge(self, l, mid, r):
        aux = self.a[l:r + 1].copy()

        i, j = l, mid + 1
        for k in range(l, r + 1):
            if i > mid:
                self.a[k] = aux[j - l]
                j += 1
            elif j > r:
                self.a[k] = aux[i - l]
                i += 1
            elif aux[i - l, self.axis] < aux[j - l, self.axis]:
                self.a[k] = aux[i - l]
                i += 1
            else:
                self.a[k] = aux[j - l]
                j += 1

    def mgsort(self, l, r):
        if l >= r:
            return
        mid = int((r + l) / 2)
        self.mgsort(l, mid)
        self.mgsort(mid + 1, r)
        self.merge(l, mid, r)
        return


class node:
    def __init__(self, val, dep):
        self.val = val[:-1]
        self.kind = val[-1]
        self.dep = dep
        self.lchild = None
        self.rchild = None


class KNeighborsClassifier:
    def __init__(self, n_neighbors=3, p=2):
        self.k = n_neighbors
        self.__p = p
        self.KDtree = None

    def set_neighbors(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, x: DataFrame, y):
        # x=x.to_numpy()
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)
        self.n, self.dim = x.shape
        self.a = np.hstack((x, y))
        self.KDtree = self.build(0, self.n - 1, 0)
        pass

    def build(self, l, r, dep):
        if l > r:
            return
        mid = (l + r) // 2
        idx = dep % self.dim

        self.a[l:r + 1] = mergesort(self.a[l:r + 1], axis=idx).a

        newnode = node(self.a[mid], dep)
        newnode.lchild = self.build(l, mid - 1, dep + 1)
        newnode.rchild = self.build(mid + 1, r, dep + 1)
        return newnode

    def query(self, t, k=None):
        if k is not None:
            self.k = k
        nearest = np.array([[float('inf'), None] for _ in range(self.k)])

        node_lst = deque()
        node = self.KDtree
        while node:
            node_lst.appendleft(node)
            dim = node.dep % self.dim
            if t[dim] <= node.val[dim]:
                node = node.lchild
            else:
                node = node.rchild

        while len(node_lst) > 0:
            node = node_lst.popleft()
            dist = np.sum(np.abs(t - node.val)**self.__p)**(1 / self.__p)

            idx_arr = np.where(dist < nearest[:, 0])[0]
            if idx_arr.size > 0:
                nearest = np.insert(nearest, idx_arr[0], [dist, node],
                                    axis=0)[:self.k]

            r = nearest[:, 0][self.k - 1]

            dim_dist = t[node.dep % self.dim] - node.val[node.dep % self.dim]

            if r > abs(dim_dist):
                append_node = node.rchild if dim_dist < 0 else node.lchild
                if append_node is not None:
                    node_lst.append(append_node)
        return np.array([n[1].kind for n in nearest])

    def predict(self, x, k=None):
        x = np.array(x).reshape(-1, self.dim)
        return np.array([
            KNeighborsClassifier.num_max(self.query(x[i], k))
            for i in range(len(x))
        ])

    def num_max(x):
        dic = Counter(x)
        return max(dic, key=dic.get)

    def score(self, x, y):
        y = np.array(y)
        x = self.predict(x)
        return sum(x == y) / len(x)


if __name__ == "__main__":
    # X = pd.DataFrame({'1': [2, 5, 9, 4, 8, 7], '2': [3, 4, 6, 7, 1, 2]})
    # Y = pd.Series([0, 0, 1, 0, 1, 1])
    X = pd.DataFrame({'1': [7, 6, 3], '2': [5, 2, 1]})
    Y = pd.Series([0, 1, 2])
    model = KNeighborsClassifier(1)
    model.fit(X, Y)
    print(model.predict(np.array([[6, 1], [5, 5], [1, 0]])))
