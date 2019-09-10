from collections import defaultdict
from math import log

#计算数组的熵 = 求和 概率*(-以2为低求概率的对数)
def calcShannonEnt(list):
    #首先计算各元素在集合中出现的次数,
    # labelCounts类型为字典dict
    labelCounts = {}
    for elements in list:
        if elements not in labelCounts.keys(): labelCounts[elements] = 0
        labelCounts[elements] +=1
    print('各元素在集合中出现的次数',labelCounts)
    #元素数量
    numEntries = len(list)
    #计算数组的熵
    shannonEnt = 0.0
    for key in labelCounts:
        #概率
        prob = float(labelCounts[key])/numEntries
        #求和 概率*(-以2为底求概率的对数)
        shannonEnt -= prob * log(prob,2)    
    return shannonEnt
    
#计算多个数据集合的熵之和
def calcShannonEnt1(list):
    # 首先计算各元素在集合中出现的次数,
    # labelCounts类型为字典dict
    labelCounts = defaultdict(int)
    for k in list:
        labelCounts[k] += 1
    print('各元素在集合中出现的次数',labelCounts)
    # 元素出现的总数量
    numEntries = len(list)
    # 计算数组的熵
    shannonEnt = 0.0
    for value in labelCounts.values():
        # 每个元素出现的概率
        prob = float(value) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
#测试
list = [1,1,0,0,0]
print(list,calcShannonEnt(list))
print(list,calcShannonEnt1(list))
print(len(list))

list = [1,1,0,0]
print(list,calcShannonEnt(list))
print(list,calcShannonEnt1(list))

list = [1,1,0]
print(list,calcShannonEnt(list))
print(list,calcShannonEnt1(list))