from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import datasets,svm
import sklearn.preprocessing as pre
from sklearn.impute import SimpleImputer
import numpy as np

# 需要采集哪些数据
# 研究是跨领域的，不应该了解数据标签，而应该通过具体数据来找到之间的关联关系，提取特征，跳出问题本身
def main():
    iris =  datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names  # 分类类别
    # print(class_names)
    feature_names = iris.feature_names  # 特征维度
    # print(feature_names)
    # des = iris.DESCR
    # print(des)
    x_train,x_test,y_train,y_test = train_test_split( X, y, random_state=0, train_size=0.7)
    classifier = svm.SVC(kernel = 'linear',C = 0.01)
    y_pred = classifier.fit(x_train,y_train).predict(x_test)

    con_m = confusion_matrix(y_test,y_pred)

    scaler = pre.Normalizer(norm='l1')
    scaler_np = scaler.fit_transform(con_m)
    print(scaler_np)

    print(accuracy_score(y_pred,y_test))
    sim = SimpleImputer(np.nan,'most_frequent',)
    X = [[np.NAN,2,3],[4,np.NAN,5],[8,3,np.NAN]]
    print(sim.fit_transform(X))
    print(X)

def test():
    # num = np.random.randint(0,10,20)
    # num2 = np.hstack([np.zeros(10,dtype=np.bool),np.ones(10,dtype=np.bool)])
    # np.random.shuffle(num2)
    # print(num2)
    # print(np.where(num2))
    a = np.logspace(0,9,12,base=2)
    print(a)


if __name__ == '__main__':
    test()