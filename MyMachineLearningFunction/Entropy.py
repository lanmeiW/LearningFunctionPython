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
   return

#测试
list = [1,1,0,0,0]
print(list,calcShannonEnt(list))
print(len(list))

list = [1,1,0,0]
print(list,calcShannonEnt(list))


list = [1,1,0]
print(list,calcShannonEnt(list))
