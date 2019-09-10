from collections import defaultdict
import numpy as np
s=[1,1,2,2,2,3,1,1,3,2,2]

for i, element in enumerate(s):
    print(i,element)

dict=defaultdict(list)
for k,v in [(v,i)for i, v in enumerate(s)]:
    dict[k].append(v)
print('dict:',dict)
#为字段设置字段名称
data = np.array([[1,2,3],[4.0,5.0,6.0],[11,12,12.3]])
a = np.array(data,dtype= {'names': ['1st','2nd','3rd'],'formats':['f8','f8','f8']})
print (a['1st'])

#测试数据集
dataSet=[[1,1,'yes'],
         [1,1,'yes'],
         [0,1,'no'],
         [1,0,'no'],
         [1,0,'no']]
labels=['技能','态度']
typeList=[]
for val in dataSet:
    for v in val:
        typeList.append(type(v))
print(typeList)
data=np.array(dataSet,dtype={'names':['技能','态度','分类结果'],'formats':[str,str,str]})
print(data)
print(data['技能'])

# zhushi

