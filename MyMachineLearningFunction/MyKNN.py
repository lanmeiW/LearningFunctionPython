#KNN算法的三个基本要素：k值的选择、距离度量和分类决策规则(常用的是多数表决规则)
#主要思想：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，
# 这k个实例的多数属于某个类，就把该输入实例分为这个类。

import math

import numpy
import pandas
class KNNMethod:
    #初始化
    def __init__(self, k,p):
        self.k = k
        self.p=p
    # fit()方法
    def fit(self,train_x,train_y):
        self.train_x = train_x
        self.train_y = train_y
    #predict()方法
    def predict(self,test_x):
        #distanceList[]记录每一个测试数据xi分别和所有训练数据的距离
        distanceList=[]
        for xi in test_x:
            tempList = []
            for xj in self.train_x:
                tempList.append(self.distance(xi, xj))
            distanceList.append(tempList)
        #取得每一行的排序索引从小到大，最后截取前k个最小的距离，即
        indexDistaince=numpy.argsort(distanceList,axis=1)[:,:self.k]
        #对应k个选取的训练集的y所属类别
        resultList=[]
        for indexD in indexDistaince:
            temp=[]
            for index in indexD:
                temp.append(self.train_y[index][0])
            resultList.append(temp)
        #计算出每一行类别最多的值,即预测的所属类别
        result=[]
        for line in resultList:
            result.append(numpy.argmax(numpy.bincount(line)))
        return result
    # Distance()方法，距离度量:输入两个点x1和x2,以及距离的选择p
    def distance(self,x1, x2):
        return math.pow(sum(map(lambda a_b: math.pow(abs(a_b[0]-a_b[1]),self.p), zip(x1, x2))), 1 / self.p)
    # 当p=1时，称为曼哈顿距离（Manhattan distance）
    def manhattanDistance(self,x1, x2):
        return sum(map(lambda a_b: abs(a_b[0]-a_b[1]), zip(x1, x2)))
    # 当p=2时，称为欧式距离（Euclidean distance）
    def euclideanDistance(self,x1, x2):
        return math.pow(sum(map(lambda a_b: math.pow(a_b[0] - a_b[1], 2), zip(x1, x2))), 1 / 2)
    #当p=无穷时，它是各个坐标距离的最大值

#读取训练数据
train_x=pandas.read_csv('D:/workspace/DataSets/doctor/train_X.csv')
train_y=pandas.read_csv('D:/workspace/DataSets/doctor/train_y.csv')
#测试数据集
test_x=pandas.read_csv('D:/workspace/DataSets/doctor/test_X.csv')
#KNN类的实例化
knn=KNNMethod( 3,2)
#fit()
knn.fit(train_x.values,train_y.values)
result=knn.predict(test_x.values)
#测试数据的真值
test_y=pandas.read_csv('D:/workspace/DataSets/doctor/test_y.csv')
print('result',result)
print('test_y',test_y)
#预测结果显示
labels=['女娃','男孩','没有怀孕']
i=0
#预测正确的数据量
predictOkNum=0
print("编号，诊断值，实际值")
while i<test_y.shape[0]:
    if result[i]==test_y.values[i,0]:
        predictOkNum+=1
        okOrNo='准确'
    else:
        okOrNo='错误'
    print("%s,%s,%s,%s" % (i+1, labels[result[i]],labels[test_y.values[i, 0]],okOrNo))
    i=i+1
print("诊断正确率为：%s" % (predictOkNum/i))






