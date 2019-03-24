from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op


# 可以看出分类和回归在类的定义上基本相同

class Net_Classify(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 初始化数据，和创建层
        super(Net_Classify, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)  # 隐藏全链接层
        self.out = nn.Linear(n_hidden, n_output)  # 输出全链接层

    def forward(self, x):  # 定义数据传递的顺序
        x = self.hidden(x)  # 数据传入第一层
        x = F.relu(x)  # activate function（总共就几个）
        x = self.out(x)  # 数据传入输出层
        return x


class Net_Regression(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 初始化数据，和创建层
        super(Net_Regression, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)  # 隐藏全链接层
        self.predict = nn.Linear(n_hidden, n_output)  # 输出全链接层

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x


def regression():
    x = torch.linspace(-1,1,100)
    x = torch.unsqueeze(x,dim=1) # 改成二维形式
    y = x**2 + 0.2 * torch.rand(x.size())  # 添加扰动因子
    # plt.scatter(x.data.numpy(),y.data.numpy())
    # plt.show()

    net = Net_Regression(n_feature=1,n_hidden=10,n_output=1)

    # 快速搭建方式(将数据处理的顺序告知)
    net2 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),  # 这里要改成无参数的形式
        nn.Linear(10, 1)

    )

    optimizer = op.SGD(net2.parameters(), lr=0.02) # 随机梯度下降
    loss_fun = nn.MSELoss()  # L2范式误差

    # 可以看出遍历过程也是相似的
    for i in range(2000):
        p = net2(x)
        loss = loss_fun(p,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)


def classify():

    d = torch.ones(100,2)
    x0 = torch.normal(-2*d,1)
    y0 = torch.zeros(100)
    x1 = torch.normal(2*d,1)
    y1 = torch.ones(100)

    x = torch.cat((x0,x1),0).type(torch.FloatTensor)
    y = torch.cat((y0,y1),0).type(torch.LongTensor)

    # plt.scatter(x[:,0].data.numpy(),x[:,1].data.numpy(),c=y.data.numpy(),cmap='RdYlGn')
    # plt.show()
    net = Net_Classify(n_feature=2,n_hidden=10,n_output=2)  # 输入层个数由数据维度决定， 输出层个数由分类类别数决定

    optimizer = op.SGD(net.parameters(),lr=0.02)  # 对梯度数据进行优化的优化器，将net的梯度值传入，后面是学习率
    loss_fun = nn.CrossEntropyLoss()  # 交叉煽误差，计算两者误差，用于反向传递

    for i in range(2000):
        out = net(x)  #将数据交给nn
        loss = loss_fun(out,y)  # 真实值必须在后面

        optimizer.zero_grad()  # 清空grad，这是由SGD的属性决定的，因为SGD通过不断迭代来累积变化值，
        loss.backward()  # 将loss反向传递
        optimizer.step()  # 优化梯度值 ，直接将获取到的梯度进行不断的迭代运算

    print(loss)


# 保存和导入
def save_load():

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )

    torch.save(net,"net.pkl")  # 保存框架和参数
    net1 = torch.load("net.pkl")  #导入框架和参数
    torch.save(net.state_dict(),"net_params.pkl")  # 只保留框架参数

    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net3.load_state_dict(torch.load("net_params.pkl"))  # 这种方式下就必须要自己先构建相似的结构，然后在导入数据


if __name__ == '__main__':
    # main()
    # test()
    regression()