import numpy as np



list = [[1,1,3,5,5,7,9,9,6],
[1,1,3,5,5,7,9,9,6],
[1,1,3,5,5,7,9,9,6],
[1,1,3,5,5,7,9,9,6],
]

print(list)
rule=[0,6,2,3,4,5,1,7]
rule.append(len(list[0])-1)
temp = np.array(list)
rule=np.array(rule)
print
newList = temp[:,rule.argsort()]
print(newList)

#calcShannonEnt()方法，计算多个数据集合的熵之和
def calcShannonEnt(list):
    # 首先计算各元素在集合中出现的次数,
    # labelCounts类型为字典dict
    labelCounts = defaultdict(int)
    for k in list:
        labelCounts[k] += 1
    print('各元素在集合中出现的次数', labelCounts)
    # 元素出现的总数量
    numEntries = len(list)
    # 计算数组的熵
    shannonEnt = 0.0
    for value in labelCounts.values():
        # 每个元素出现的概率
        prob = float(value) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
#计算每个属性的熵
entropyList=[]
#首先计算每一类熵的值（比如该特征分为0，1两类，）
for i in range(featureNum):
    print('第i列的计算输出：',i)
    #获得每个属性列
    feature=np.array(dataSet)[:,i]
    #标签列
    y=np.array(dataSet)[:, -1]
    #该列的熵
    entropy=0
    print(y)
    #featureDict: defaultdict(<class 'list'>, {'1': [0, 1, 2], '0': [3, 4]})
    featureDict=defaultdict(list)
    for k,v in [(v,i)for i,v in enumerate(feature)]:
        featureDict[k].append(v)
    print(len(featureDict))
    print('featureDict：',featureDict)
    yLists=[]
    for k,v in featureDict.items():
        print(k, v)
        # 1[0, 1, 2]
        # 0[3, 4]
        #对应索引号的y标签：yList: ['yes', 'yes', 'no']
        yList=[(val) for i, val in enumerate(y) if i in v]
        yLists.append(yList)
        print('yList:',yLists)
        #熵
        entropy=entropy+calcShannonEnt(yList)
        print('entropy:',entropy)
    entropyList.append(entropy)
print('entropyList:',entropyList)
#根据信息熵将各个属性feature进行排序，从小到大，熵越小的越排在前面
print(sorted(range(len(entropyList)), key=lambda k: entropyList[k]))

class TreeNode:
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None