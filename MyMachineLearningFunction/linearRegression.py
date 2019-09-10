from numpy import *
#题目中数据
X=mat([[40,16,27,50],[20,39,2,47]])
y=mat([[33],[14]])
#利用最小二乘法求得w
w=(X.T*X).I*X.T*y
print('w',w)
#x*w
x=[10,9,36,16]
y_predict=x*w
print('y_predict',y_predict)

#只有数据量大于特征数据维度，才有效
#数据量大于特征数据的维度，才可以计算得到w
X=mat([[40,16,27,50],[20,39,2,47],[56,45,88,57],[18,78,16,9]])
y=mat([[33],[14],[132],[103]])

w=(X.T*X).I*X.T*y
print('w',w)

#预测
x=[10,9,36,16]
y_predict=x*w
print('y_predict=x*w',y_predict)

#加载文本文件中数据
def loadDataSet(fileName):
    #打开文件
    fr = open(fileName)
    #特征数
    numFeat = -1
    #特征值数组
    Xlist = []
    #结果数组
    ylist = []
    for line in fr.readlines():
        #取得一行特征lineArr
        lineArr = []
        #用制表符分拆
        curLine = line.strip().split('\t')
        if(numFeat==-1):
            numFeat = len(curLine)-1
        #将特征转换为float类型
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        #添加到数组中
        Xlist.append(lineArr)
        ylist.append(float(curLine[-1]))
    #结果转换为矩阵返回
    return Xlist,ylist

#main()函数，进行测试
if __name__=='__main__':
    # 读取数据
    Xlist, ylist = loadDataSet('D:/workspace/DataSets/linearRegressione/x0.txt')
    # 转换为矩阵
    X, y = mat(Xlist), mat(ylist).T
    w = (X.T * X).I * X.T * y
    # 图形化显示
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制原始数据
    ax.scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0])
    # 绘制模型图
    xCopy = X.copy()
    # 将点按照升序排列
    xCopy.sort(0)
    yPredict = xCopy * w
    ax.plot(xCopy[:, 1], yPredict)
    plt.show()
    # 实际与预测之间的相关性
    corrcoef((X * w).T, y.T)

    # 利用梯度下降法实现 见下一节课










