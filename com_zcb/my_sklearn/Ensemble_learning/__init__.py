from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification,make_moons,make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.axes import Axes
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda ,QuadraticDiscriminantAnalysis as qda



import numpy as np
def main():
    names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM',
             'Decision Tree','Random Forest','AdaBoost'
             # ,'Naive Bayes','LDA','QDA'
             ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel='linear',C=0.025),
        SVC(gamma=2,C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
        AdaBoostClassifier(),
        # GaussianNB(),
        # lda(),
        # qda()

    ]
    X,y = make_classification(n_samples=200,n_features=2,n_classes=2,n_informative=2,n_redundant=0,
                              random_state=1, n_clusters_per_class=1)
    np.random.seed(1)
    X+=2*np.random.uniform(size=X.shape) # add random error
    classification_data = (X,y)
    datasets = [
        make_moons(noise=0.3,random_state=0),
        make_circles(noise=0.2,random_state=1),
        classification_data]

    cm_bright = ListedColormap(['#FF0000','#0000FF']) # 颜色板设计
    cm_bright2 = ListedColormap(['#FFdd00','#00ddFF']) # 颜色板设计
    plt.figure(figsize=(27,9))  # 画布的大小
    i = 1
    for ds in datasets:
        # 基本数据处理
        X,y = ds  # X有两个属性
        X = StandardScaler().fit_transform(X)
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
        x_min,x_max = X[:,0].min()-.5 , X[:,1].max()+.5
        y_min,y_max = X[:,1].min()-.5 , X[:,1].max()+.5

        # 求xx,yy的目的是要画出整个图的等高线
        xx,yy = np.meshgrid(np.arange(x_min,x_max,.02), np.arange(y_min,y_max,.02))

        plt.subplot(len(datasets),len(classifiers)+1,i)
        # 画基本数据图
        plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_bright, s=80) # 画出训练样本
        plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_bright2, alpha=0.5,s=80)  # 画出测试样本，通过透明度区分
        i+=1
        plt.xlim(xx.min(),xx.max())
        plt.ylim(yy.min(),yy.max())
        plt.xticks()  # 设置x坐标轴的显示情况，空参数就是关闭这个功能
        plt.yticks()
        for name,clf in zip(names,classifiers):
            plt.subplot(len(datasets),len(classifiers)+1,i)
            clf.fit(x_train,y_train)
            score = clf.score(x_test,y_test)  # 将训练的模型直接用于测试集中，并计算测试集得分

            if hasattr(clf,"decision_function"): # 判断是否有这个属性
                z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
            else:
                z = clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1] # 生成的是每个分类的概率，二分类，取其中一列的概率做等高线

            z = z.reshape(xx.shape) # 预测了每个坐标点的分类（这里不是样本的分类）
            plt.contourf(xx,yy,z,cmap = cm_bright,alpha=.8) # 绘制等高线图（热量图）
            i+=1



    plt.show()





    # z = np.sqrt(x**2+y**2)
    # z = [1,2,3,4,4,3,4,4,334,664,668,1000]
    # x = np.arange(1,13)
    # y = np.arange(1,13)
    # print(z)
    # plt.subplot(3,4,1)
    # plt.scatter(x,y,s=80,c=z,cmap=cm_bright)
    # plt.subplot(3, 4, 1)
    # plt.scatter(x, y, s=80, c=z, cmap=cm_bright)
    # plt.show()


def test():
    xx,yy = np.meshgrid(np.arange(0,10,.02),np.arange(0,10,.02))
    print(xx)





if __name__ == '__main__':
    main()