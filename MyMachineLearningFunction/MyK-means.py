import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KmeansClustering:
    #聚类属于无监督学习
    #初始化:k和p
    # k为所分类别数目，
    # p为距离选择（当p=1时，称为曼哈顿距离（Manhattan distance）；当p=2时，称为欧式距离（Euclidean distance）
    def __init__(self,k,p):
        self.k=k
        self.p=p
    #fit()方法，
    def fit(self,data):
        #self.data=data
        self.n=len(data[0])
        #初始化k个中心点，随机选择
        row_rand_array = np.arange(data.shape[0])
        np.random.shuffle(row_rand_array)
        self.centerList = data[row_rand_array[0:self.k]]
        #后去初始中心点
        self.center1 =np.array(self.center(data))
        self.centerList =self.center1
        print('self.centerList=:',self.centerList)
        # 将所有数据集分到k个类中
        y = self.predict(data)
        i=0
        while True:
            y1=y
            # 将每一类添加到同一行，
            #结果类似这样：defaultdict(<class 'list'>, {0: [0，4，6，7], 1: [1，3，8], 2: [2，5，9，10]})
            classifyDict = defaultdict(list)
            for k, va in [(v, i) for i, v in enumerate(y1)]:
                classifyDict[k].append(va)
            # 重新计算中心点
            self.centerList = []
            for j in range(self.k):
                # 同一类中的所有数据
                classList = [data[x] for x in classifyDict[j]]
                print ('classList type:=',type(classList))
                self.centerList.append(self.newCenter(classList))
            # 重新分类
            y = self.predict(data)
            i=i+1
            print('i=:',i)
            print('self.centerList=:',self.centerList)
            if y==y1:
                break
    #predict()方法，——————根据聚类中心点预测每个数据所属的类别，实现分类
    def predict(self,X):
        # y记录所属类别：0至（k-1）,共k个类
        #误差平方和
        self.SSE=0
        y=[]
        for val in X:
            tempList = []
            #将val分到k个中心点中距离最近的类别中
            for i in range(self.k):
                tempList.append(self.distance(val,self.centerList[i]))
                i=i+1
            self.SSE = self.SSE+min(tempList)
            y.append(tempList.index(min(tempList)))
        return y
    #Center()方法，——————获取初始中心点
    def center(self,data):
        firstCenter=[]
        temp=data[0]
        for i in range(1, len(data[0])):
            t=[]
            for j in range(0, len(data[0])):
                val=temp[j] + data[i][j]
                t.append(val)
            temp = t
        for i in range(1,self.k+1):
            firstCenter.append(np.array(temp)*i/(self.k+1))
        return firstCenter
    #newCenter()方法，——————重新获得聚类中心点
    def newCenter(self, classList):
        temp = classList[0]
        for i in range(1, len(classList[0])):
            t = []
            for j in range(0, len(classList[0])):
                val = temp[j] + classList[i][j]
                t.append(val)
            temp = t
        return np.array(temp) / self.k
    # Distance()方法，——————距离度量:输入两个点x1和x2,以及距离的选择p
    def distance(self, x1, x2):
        return math.pow(sum(map(lambda a_b: math.pow(abs(a_b[0] - a_b[1]), self.p), zip(x1, x2))), 1 / self.p)


#K-means方法测试
#数据准备
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

#建立模型
kmeans=KmeansClustering(3,2)
#训练
kmeans.fit(X)
#预测
y_pred=kmeans.predict(X)
print('SSE2：',kmeans.SSE)
#结果显示
plt.figure(figsize=(12, 12))
print('y_pred:',y_pred)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")


plt.subplot(222)
plt.scatter(kmeans.center1[:, 0], kmeans.center1[:, 1])
plt.title("Incorrect Number of Blobs")
plt.show()

