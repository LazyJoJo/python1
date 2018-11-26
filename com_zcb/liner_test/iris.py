from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# 基本的点图查看
def see_data(data,target,loss,W):


    # X = data[0:150,[0,3]]  # 与0搭配时，1，3具有相同的特性，2具有特殊特性
    # Y = target[0:150]
    # label = np.array(Y)
    # set_label = set(label)
    # for i in set_label:
    #     print(i,len(np.where(label==i)[0]))
    # print(np.where(label==2))
    # index_0 = np.where(label==0) # 获得值为0的索引
    index_0 = np.where(target==0)
    plt.scatter(data[index_0,0], data[index_0,1], marker='x',color = 'b', label = '0')
    index_1 = np.where(target==1)
    plt.scatter(data[index_1,0],data[index_1,1],marker='o',color='r',label = '1' )
    plt.xlabel('X0')
    plt.ylabel('X2')
    plt.legend(loc='upper left')  #显示标签的位置
    plt.show()
    plt.plot(loss)
    plt.show()


def train_data(X,y,learn_rage = 0.01 , num_iters = 5000):
    num_train,num_feature = X.shape  # 训练数和每个样本的维度
    W = 0.001*np.random.randn(num_feature,1)  # 生成服从正太分布的随机向量(n*1)
    # 每次学习都是以所有样本一起学习的，即n个样本学习一次W，求得是最大值
    loss_home = []
    for i in range(num_iters):
        loss = compute_loss(X,y,W)
        h = sigmod(X@W)

        dw = np.dot(X.T,(y-h))/num_train
        W = W+learn_rage*dw  # 这里将公式变形为乘以每一维度的平均值（注意是每一维度，不是每个样本），这是为了快速迭代n各样本
        loss_home.append(loss)

    return W,loss_home

def compute_loss(X,y,W):
    # X mat(n,m)  y ndarray(n,1) W ndarray(m,1)
    h = sigmod(X@W)

    train_num = len(y)
    loss = np.sum(y*np.log(h)+(1-y)*np.log(1-h))/train_num  #损失函数只是用来记录看是否迭代平稳
    return loss

def sigmod(num):
    return 1/(1+np.exp(-num))


def main():
    print("main")
    # 逻辑回归的特点：
    #
    # 模型训练完后可以直接使用，但模型将无法更改
    # 迭代次数无法确定，
    # 分类只能是2分类，
    # 最优化时，其能力取决于所使用的算法的能力，这里使用的梯度下降法解最优化问题，因此会存在梯度下降的各种问题
    # 解释1，为何需要用到梯度下降，因为偏导数等于0的解求不出来，所以只能用梯度下降，慢慢逼近最优解
    # 解释2，迭代时dw为何使用的是平均值，原因是方便计算，可以在一次迭代中用矩阵计算所有样本数据，加快迭代速度，本来总的迭代数是10000(迭代数)*100(样本数)，现在只要10000次计算就好
    # 解释3，为何每个样本就要迭代一次，主要是由于梯度下降算法的特性
    #
    # 解题过程：列出表达式，改为log级别，求出损失函数（正好是其概率函数，这里用到最大似然估计，不能用最小二乘法，因为逻辑回归方程不是线性方程），对损失函数求导求出最大或最小值，用梯度下降法解最大最小值

    iris = load_iris()
    data = iris.data
    target = iris.target
    X = data[0:100,:] # 改为使用全部维度
    # 从观测图中可以看出，单纯使用两个维度都不足以区分两个样本集，都会出现误判情况（某一群落中出现别的群落个别点的情况），需要全部4个维度
    o = np.ones((100,1))
    X = np.hstack((o,X)) # 方程是w0+w1*x1+w2*x2,所以X要加一个维度，所有样本的该维度都是1
    print(np.mat(X).shape)
    y = target[:100]
    y = np.reshape(y,(-1,1))
    print(y.shape)
    W,loss = train_data(X,y,num_iters=10000)
    print("loss------------------")
    print(loss)
    print("w---------------------")
    print(W)
    X = X[:,1:]
    see_data(X,y,loss,W)  # W为最终模型参数  loss为最终偏差量




if __name__ == '__main__':
    main()
    # see_data()


