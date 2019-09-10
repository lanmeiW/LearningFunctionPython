
import pandas
from sklearn.tree import DecisionTreeClassifier
#读取训练数据集
train_x=pandas.read_csv('D:/workspace/DataSets/doctor/train_X.csv')
train_y=pandas.read_csv('D:/workspace/DataSets/doctor/train_y.csv')
#创建机器人医生
doctor=DecisionTreeClassifier()
#训练机器人医生
doctor.fit(train_x,train_y)
#测试机器人医生，并得到预测结果result数组
test_x=pandas.read_csv('D:/workspace/DataSets/doctor/test_X.csv')
result=doctor.predict(test_x)
#打印诊断结果
test_y=pandas.read_csv('D:/workspace/DataSets/doctor/test_y.csv')
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


