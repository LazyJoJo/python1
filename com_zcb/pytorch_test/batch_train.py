import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

#  用于数据量很大时，将数据划分小块进行训练

def batch_device():
    BATCH_SIZE = 60  # 不一定要是x的整数倍
    x = torch.linspace(1,100,100)
    y = torch.linspace(301,400,100)

    torch_dataset = Data.TensorDataset(x,y)  # 封装到数据集中
    loader = Data.DataLoader(  # 批量数据加载器
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,   # 随机打乱数据
        num_workers=3   # 多进程来读取数据（开几个进程来加载数据）0表示只有主进程进行加载
    )

    for epoch in range(3):   # 实现多次重复随机训练
        for step ,(batch_x,batch_y) in enumerate(loader): # enumerate 会给数据加个索引step
            # 下面对每一批次的数据进行训练



            print("Epoch ",epoch,"| step",step,"| batch_x",
                  batch_x.numpy(),"| batch_y",batch_y.numpy())

def test():
    x = torch.linspace(0,100,10000)
    y = x**(0.5)
    x2 = np.linspace(0,100,10000)
    y2 = x2
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.scatter(x2,y2)
    plt.show()


if __name__ == '__main__':
    # batch_device()
    test()

