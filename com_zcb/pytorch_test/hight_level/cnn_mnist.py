import torch
import torch.nn as nn
import torchvision  # 数据库模块
import torch.utils.data as Data  # 数据批处理
import torch.optim as op
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

import ssl

ssl._create_default_https_context = ssl._create_unverified_context # 忽视证书错误


# class MyDataset(Dataset):
    # def __init__(self, image_paths, transforms=transforms):
    #     self.image_paths = image_paths
    #     self.transforms = transforms
    #
    # def __getitem__(self, index):
    #     image = Image.open(self.image_paths[index])
    #     image = image.convert('RGB')
    #     if self.transforms:
    #         image = self.transforms(image)
    #     return image


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(  # 输入前 --> (n,1,28,28) 数据条数，高，长，宽
            nn.Conv2d(
                in_channels=1,  # 原先的特征值，输入维度，当成高度
                out_channels=16,  # 希望提取出来的特征值，他这边都当成高度
                kernel_size=(5, 5),  # 筛选的区域，如果是单个数长宽都是5，如果是两个数则分别定义长高
                stride=1,  # 筛选区移动的步长 （基本都设成1，为了更好的提取信息）
                padding=2,  # 四周都补2行（caffe,TensorFlow这个参数的原理不同），想要出来的图片尺寸没变化，
                # padding的计算公式：(kernel-1)/2 只有当stride为1时才满足
            ),  # --> (n,16,28,28) 出来的尺寸，计算公式：nsize =(osize-kernel+2*padding)/stride+1
            nn.ReLU(),  # --> (n,16,28,28)
            nn.MaxPool2d(kernel_size=2),  # 池化操作2*2的区间里取最大值，相当于数据缩小了一倍 --> (n,16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),  # --> (n,32,14,14)
            nn.ReLU(),  # 不变
            nn.MaxPool2d(2)  # --> (n,32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)  # 数据展平后，分成10类

    def forward(self, data):  # 数据处理流程
        x = self.conv1(data)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # --> (n,32*7*7)展平操作，将后面的3维数据变成1维
        x = self.out(x)
        return x


def cnn_train():
    EPOCH = 1  # 数据全集运行几次，这边只运行一次
    BATCH_SIZE = 50
    LR = 0.001  # 学习率
    DOWNLOAD_EMNIST = True

    train_data = torchvision.datasets.EMNIST(
        root='./mnist/',  # 下载地址
        split='digits',
        train=True,  # 返回训练数据集，否则返回测试数据集
        transform=torchvision.transforms.ToTensor(),  # 训练时数据自动转换成【0，1】
        download=DOWNLOAD_EMNIST  # 是否下载数据
    )
    test_data = torchvision.datasets.EMNIST(root='./mnist/',split="digits",train=False)  # split 属性未知
    # print(train_data.train_data.size())
    # print(test_data.test_data.size())
    # print(train_data.train_data[0].numpy().T)  # 转置之后画出来的图像才是正确的，不知道为何
    # plt.imshow(train_data.train_data[0].numpy().T,cmap='gray')
    # plt.title("%d" % train_data.train_labels[0].numpy())
    # plt.show()

    # print(test_data.test_data[0])

    test_x = torch.unsqueeze(test_data.test_data[:2000],dim=1).type(torch.FloatTensor)/255.  # 为了和训练的数据一致，也要除255
    test_y = test_data.test_labels[:2000]

    train_load = Data.DataLoader(dataset=train_data,num_workers=2,batch_size=BATCH_SIZE,shuffle=True)

    cnn = CNN()
    loss_fun = nn.CrossEntropyLoss()  # 交叉煽计算误差
    optimizer = op.Adam(cnn.parameters(),lr=LR)

    for epoch in range(EPOCH):
        for num,(train_x,train_y) in enumerate(train_load):
            # print(train_x.size())
            out = cnn(train_x)
            loss = loss_fun(out,train_y)  # 这里的两个数据维度不相同，前一个是(n,10),后一个为(10,)其中的10为类别
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(test_x.size())
            # out2 = cnn(test_x)
            # loss2 = loss_fun(out2,test_y)

            print("step",num,"| train loss:",loss.data)

    torch.save(cnn,"./cnn.pkl")

def cnn_test():

    cnn = torch.load('./cnn.pkl')
    loss_fun = nn.CrossEntropyLoss()
    test_data = torchvision.datasets.EMNIST(root='./mnist/', split="digits", train=False)  # split 属性未知
    test_x = torch.unsqueeze(test_data.test_data[:50], dim=1).type(torch.FloatTensor) / 255.  # 为了和训练的数据一致，也要除255
    test_y = test_data.test_labels[:50].numpy()
    # print(test_y.size())

    y = cnn(test_x)
    print(y)
    pred_y = torch.max(y, 1)[1].numpy()  # 获取真实类别信息
    print(pred_y)
    count = 0
    for i in range(len(test_y)):
        if test_y[i]!=pred_y[i]:
            count+=1

    print(count/len(test_y))   # 正确率99.1%






if __name__ == '__main__':
    cnn_train()
    # cnn_test()
