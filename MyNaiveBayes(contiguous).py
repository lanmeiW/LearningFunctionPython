from collections import defaultdict
import math
import numpy as np
import pandas
#朴素贝叶斯（特征属性数据是离散的）
class GaussianDistribution:
    def __init__(self,mean,std,classSamples):
        self.mean=mean
        self.std=std
        self.classSamples=classSamples
class NaiveBayes:
    #fit()函数，进行训练得到预测模型
    def fit(self,train_X,train_y):
        #训练样本的总数
        self.sampleSum=np.array(train_X).shape[0]
        #特征总数
        self.featureSum=np.array(train_X).shape[1]
        #对训练数据train_y进行处理
        #train_y=['yes', 'yes', 'no', 'no', 'no']
        #yDict=defaultdict(<class 'list'>, {'yes': [0, 1], 'no': [2, 3, 4]})
        self.yClasses=defaultdict(list)
        for k,v in [(v,i) for i,v in enumerate(train_y) ]:
            self.yClasses[k].append(v)
        #总样本的各个分类结果的概率，y_probability={'yes': 0.4, 'no': 0.6}
        self.yClasses_probability={}
        for k,v in self.yClasses.items():
            self.yClasses_probability[k]=len(v)/self.sampleSum
        #X中所有特征的概率
        self.XFeatures_probability=defaultdict(dict)
        #按照最终类别进行划分（yes,no）两类
        for key,value in self.yClasses.items():
            # 取出train_X中想要的行数据（第1行和第2行），train_X_k= [[1 1] [1 1]]
            train_X_class= np.array(train_X)[[i for i in value]]
            classSamples=len(value)
            # 所有属性特征的概率
            features_probability = defaultdict(dict)
            #按照特征属性进行划分，分别计算第i列的属性概率
            for i in range(self.featureSum):
                #每一个特征的所有取值的概率
                feature_probability = defaultdict(int)
                # 第i列属性值,feature= [1 1]
                feature = train_X_class[:, i]
                # 求均值
                mean = np.mean(feature)
                # 求一个数组的标准差
                std = np.std(feature)
                # 求概率，其中x可以是一个数，或者一个numpy数组
                #y = np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)
                #将该列高斯分布函数的参数保存到gaussianDistribution
                gaussianDistribution=GaussianDistribution(mean,std,classSamples)
                #将第i列的高斯概率参数保存到features_probability
                features_probability[i] = gaussianDistribution
            self.XFeatures_probability[key]=features_probability
        #self.XFeatures_probability=
        #{'yes': {0（第0个属性）:gaussianDistribution,
        #         1（第1个属性）:gaussianDistribution,
        #  'no': {0（第0个属性）:gaussianDistribution,
        #         1（第1个属性）:gaussianDistribution}}
        print('self.XFeatures_probability',self.XFeatures_probability)

    # predict()函数，根据预测模型进行预测得到最终分类结果
    def predict(self,test_X):
        self.y_predict=[]
        for x in test_X:
            #self.x_predict(x)
            probability = {}
            #按照分类结果k进行划分key=yes,no
            for key, value in self.XFeatures_probability.items():
                #遍历每一个特征
                p=1
                for i in range(self.featureSum):
                    print('i',i)
                    #第i个该特征的属性值为feature_x
                    feature_x = x[i]
                    print('feature_x',feature_x)
                    #第i个该特征的高斯概率参数
                    gaussianDistribution=value[i]
                    print('gaussianDistribution',gaussianDistribution.mean,gaussianDistribution.std,gaussianDistribution.classSamples)
                    # 求概率，其中x可以是一个数，或者一个numpy数组
                    # y = np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)
                    if gaussianDistribution.std==0:
                        feature_x_probability=1
                    else:
                        feature_x_probability=np.exp(-(feature_x - gaussianDistribution.mean) ** 2 / (2 * gaussianDistribution.std ** 2)) / (math.sqrt(2 * math.pi) * gaussianDistribution.std)
                    print('feature_x_probability',feature_x_probability)
                    print('self.sampleSum',self.sampleSum)
                    feature_x_probability=feature_x_probability*gaussianDistribution.classSamples/self.sampleSum
                    p=p*feature_x_probability
                    print('p',p)
                probability[key]=p*self.yClasses_probability[key]
            temp=0
            #选择概率最大的类别作为最终分类结果
            for k,v in probability.items():
                if temp<v:
                    temp=v
                    #记录概率最大的类别
                    result=k
            self.y_predict.append(result)
        return self.y_predict


#NaiveBayes()测试程序：main()函数
if __name__ == '__main__':
    print('This is a main()函数')
    #训练数据集
    # 读取训练数据集
    train_X = pandas.read_csv('D:/workspace/DataSets/doctor/train_X.csv')
    train_y = pandas.read_csv('D:/workspace/DataSets/doctor/train_y.csv')
    #测试数据集
    test_X = pandas.read_csv('D:/workspace/DataSets/doctor/test_X.csv')
    #先转为numpy,再转为list
    train_X = np.array(train_X)  # np.ndarray()
    train_X = train_X.tolist()  # list

    train_y = np.array(train_y)  # np.ndarray()
    train_y = train_y.tolist()  # list
    train_y = [x[0] for x in train_y]

    test_X = np.array(test_X)  # np.ndarray()
    test_X = test_X.tolist()  # list
    
    #创建NaiveBayes实例
    naiveBayes=NaiveBayes()
    #训练
    naiveBayes.fit(train_X,train_y)
    #预测,得到分类结果
    result=naiveBayes.predict(test_X)
    print('result',result)

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














