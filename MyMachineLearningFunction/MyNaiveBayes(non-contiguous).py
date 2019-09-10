from collections import defaultdict
import numpy as np
#朴素贝叶斯（特征属性数据是离散的）
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
            # 所有属性特征的概率
            features_probability = defaultdict(dict)
            #按照特征属性进行划分，分别计算第i列的属性概率
            for i in range(self.featureSum):
                #每一个特征的所有取值的概率
                feature_probability = defaultdict(int)
                # 第i列属性值,feature= [1 1]
                feature = train_X_class[:, i]
                # 将第i列属性值进行分类,featureDict= defaultdict(<class 'list'>, {1: [0, 1]})
                featureDict = defaultdict(list)
                for k, v in [(element, i) for i, element in enumerate(feature)]:
                    featureDict[k].append(v)
                #遍历该特征的每一个可能值，并计算其概率
                for k, v in featureDict.items():
                    feature_probability[k]=(len(v)+1)/(self.sampleSum+1)
                features_probability[i] = feature_probability
            self.XFeatures_probability[key]=features_probability
        #self.XFeatures_probability=
        #{'yes': {0（第0个属性）:{1: 0.5},
        #         1（第1个属性）:{1: 0.5}},
        #  'no': {0（第0个属性）:{0: 0.3333333333333333, 1: 0.5},
        #         1（第1个属性）:{1: 0.3333333333333333, 0: 0.5}}}
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
                    #第i个该特征的属性值为feature_x
                    feature_x = x[i]
                    #该属性值为k类别中的概率
                    feature_x_probability=value[i][feature_x]
                    #若为0，则feature_x_probability=1/self.sampleSum
                    if feature_x_probability==0:
                        feature_x_probability=1/self.sampleSum
                    p=p*feature_x_probability
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
    train_X=[[1, 1],
          [1, 1],
          [0, 1],
          [1, 0],
          [1, 0]]
    train_y=['yes', 'yes', 'no', 'no', 'no']
    labels=['技能', '态度']
    #测试数据集
    test_X=[
          [1, 0]]
    #创建NaiveBayes实例
    naiveBayes=NaiveBayes()
    #训练
    naiveBayes.fit(train_X,train_y)
    #预测,得到分类结果
    y_predict=naiveBayes.predict(test_X)
    print('y_predict',y_predict)














