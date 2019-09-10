from collections import defaultdict
from math import log
import pprint
import copy

import numpy as np
import pandas as pd


# TreeNode类，定义树的结点
class TreeNode:
    def __init__(self, x):
        self.val = x
        # self.nodeDict = defaultdict(TreeNode)
        # k为该属性值，v为k相应的下一个结点结点node
        self.nodeDict = {}
    # DecisionTrees类，决策树类


class DecisionTrees:
    # 决策树
    # 参数初始化，
    def __init__(self):
        return

    # fit()方法，
    def fit(self, train_x, train_y):
        # 构建决策树node
        self.node = self.getTree(train_x, train_y)
        #将树变成字典格式
        print('将树变成字典格式:')
        nodeCopy = copy.deepcopy(self.node)
        self.dictTree=self.treeToDict(nodeCopy)
        print('dictTree',self.dictTree)
        # self.preOrderTree(self.node)

    # predict()方法，
    def predict(self, test_x):
        self.y_predict = []
        nodeCopy1 = copy.deepcopy(self.node)
        for i in range(len(test_x)):
            x = test_x.iloc[[i]]
            self.predictTree(nodeCopy1, x,i)
        return self.y_predict

    # predictTree（）方法，决策树预测
    def predictTree(self, node, x,i):
        #读取属性node.val的值
        key = x.loc[i, node.val]
        result=node.nodeDict[key]
        if type(result)==TreeNode:
            self.predictTree(result, x,i)
        else:
            #将结果加到y_predict
            self.y_predict.append(result)

    # TreeToDict，将树转化为字典类型并返回
    def treeToDict(self, node):
        #创建字典treeDict{}
        treeDict = {}
        #构建字典
        treeDict[node.val] = node.nodeDict
        #遍历字典
        for k, v in treeDict[node.val].items():
            #如果字典中的V为TreeNode类型，则需要把其重新变成字典并返回
            if type(v)==TreeNode:
                treeDict[node.val][k]=self.treeToDict(v)
        return treeDict

    # 打印决策树
    def preOrderTree(self, node):
        print(node.val)
        if (node.nodeDict == {}):
            return
        for k, v in node.nodeDict.items():
            print('k=', k)
            if type(v) == TreeNode:
                print(v)
                self.preOrderTree(v)
            else:
                print('v=', v)

    # 创建决策树
    def getTree(self, train_x, train_y):
        # 分类终止条件：2种情况（任意一种情况出现都可终止）
        # 情况1：如果该类中的数据，最后标签只有一种，则分类结束，此叶子结点为分类结果
        if (train_y.values == train_y.values[0]).all():
            return train_y.values[0].tolist()
        # 情况2：并没有统一分到一类中，且没有其他属性可以继续分类，则将出现可能性最大的一类做为最后分类结果
        elif train_x.empty:
            # 对应索引号的y标签：yList: ['yes', 'yes', 'no']
            yList = train_y[0].tolist()
            yListDict = defaultdict(list)
            for k, v in [(v, i) for i, v in enumerate(yList)]:
                yListDict[k].append(v)
            temp = 0
            keyMore = 0
            for k, v in yListDict.items():
                # 记录样本最多的类别
                if temp < len(v):
                    temp = len(v)
                    keyMore = k
            return keyMore
        else:
            minEntropyIndex = self.featureSort(train_x, train_y)
            # 以第minEntropyIndex列属性作为分类属性
            node = TreeNode(list(train_x)[minEntropyIndex])
            # 获取df_dataSet的第minEntropyIndex列的数据
            feature = train_x.values[:, minEntropyIndex]
            # featureDict: defaultdict(<class 'list'>, {'1': [0, 1, 2], '0': [3, 4]})
            # 根据该列的所有属性值进行分类到一个字典，将所有属性值作为k,相应属性值的所有数据的索引号的集合为v
            featureDict = defaultdict(list)
            for k, v in [(v, i) for i, v in enumerate(feature)]:
                featureDict[k].append(v)
            for key, values in featureDict.items():
                # 1[0, 1, 2]    0[3, 4]
                # 重建新的子数据(训练数据集的输入和输出)
                rows = values  # 定义行数
                columns = [i for i in range(len(train_x.values[0])) if i != minEntropyIndex]  # 定义列数
                # 获取新的指定行列的训练数据
                childTrain_x = train_x.iloc[[i for i in rows], [j for j in columns]]
                childTrain_y = train_y.iloc[[i for i in rows]]
                # 将子节点添加到node的nodeDict字典中
                node.nodeDict[key] = self.getTree(childTrain_x, childTrain_y)
        return node

    # featureSort()方法，根据信息熵将各个属性feature进行排序，从小到大，熵越小的越排在前面
    # 计算各个列的熵，得到最小信息熵的索引号
    def featureSort(self, train_x, train_y):
        # 计算每个属性的熵
        entropyList = []
        # 首先计算每一类熵的值（比如该特征分为0，1两类，）
        for i in range(train_x.shape[1]):
            # 获得每个属性列
            feature = train_x.values[:, i]
            # 该列的熵
            entropy = 0
            # featureDict: defaultdict(<class 'list'>, {'1': [0, 1, 2], '0': [3, 4]})
            featureDict = defaultdict(list)
            for k, v in [(v, i) for i, v in enumerate(feature)]:
                featureDict[k].append(v)
            for k, v in featureDict.items():
                # 1[0, 1, 2]
                # 0[3, 4]
                # 对应索引号的y标签：yList: ['yes', 'yes', 'no']
                yList = (train_y.iloc[[i for i in v]])[0].tolist()
                # 计算熵，并将熵累加
                entropy = entropy + self.calcShannonEnt(yList)
            entropyList.append(entropy)
        # 根据信息熵将各个属性feature进行排序，从小到大，熵越小的越排在前面
        self.entropyListSortIndex = sorted(range(len(entropyList)), key=lambda k: entropyList[k])
        # 得到最熵最小列的序号,并返回结果minEntropyIndex
        # print('minEntropyIndex',entropyList)
        minEntropyIndex = entropyList.index(min(entropyList))
        return minEntropyIndex

    # calcShannonEnt()方法，计算多个数据集合的熵之和
    def calcShannonEnt(self, listData):
        # 首先计算各元素在集合中出现的次数,
        # labelCounts类型为字典dict
        labelCounts = defaultdict(int)
        for k in listData:
            labelCounts[k] += 1
        # print('各元素在集合中出现的次数', labelCounts)
        # 元素出现的总数量
        numEntries = len(listData)
        # 计算数组的熵
        shannonEnt = 0.0
        for value in labelCounts.values():
            # 每个元素出现的概率
            prob = float(value) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

# 测试数据集
# 输入数据
data_x = [[1, 1,1],
          [1, 1,1],
          [0, 1,0],
          [1, 0,0],
          [1, 0,2]]
# 输出数据
data_y = ['yes', 'yes', 'no', 'no', '待定']
# 输入数据的标签
labels = ['技能', '态度','青星大学']
# 训练集:输入树和输出输出数据
# 转化为pandas的DataFrame格式的数据
train_x = pd.DataFrame(data_x)
# 为训练集的输入数据加上列名称标签
train_x.columns = labels
train_y = pd.DataFrame(data_y)
print('train_x:', train_x)
print('train_y:', train_y)
#测试数据
test= [[1, 1,1],
          [1, 1,1],
          [0, 1,0],
          [1, 0,0],
          [1, 0,2]]
# 转化为pandas的DataFrame格式的数据
test_x = pd.DataFrame(test)
# 为训练集的输入数据加上列名称标签
test_x.columns = labels

#创建决策树对象
decisionTrees = DecisionTrees()
#训练并输出决策树
decisionTrees.fit(train_x, train_y)
#测试，并输出预测结果
y_predict=decisionTrees.predict(test_x)
print('y_predict=',y_predict)
#