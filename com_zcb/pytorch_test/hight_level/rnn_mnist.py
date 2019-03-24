import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.optim as op

EPOCH = 1
BATCH_SIZE = 50
TIME_STEP = 28  # RNN时间步数，图片size的高度  将图片理解成时间序列
IMPUT_STEP = 28  # RNN输入数据，图片size的长度
LR = 0.01
DOWNLOADS = False


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(  # lstm比RNN优秀很多，这里使用RNN无法收敛
            input_size=28,  # 这个是由数据决定的
            hidden_size=64,  # 这个数据不知道如何定
            num_layers=1,  # 隐藏层数量，数量越多能力越强，但是时间也越久
            batch_first=True  # (batch,time_step,input)  设定数据的组成方式
        )
        self.out = nn.Linear(64,10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (num_layers, batch, hidden_size)
        # h_c shape (num_layers, batch, hidden_size)
        # r_out是结果，剩下两个是迭代时产生的数据，h_n（分线程数据）h_c(主细胞数据)
        # None 值表示第一次迭代时，是否有前置的隐藏层信息，有的话传入数据，没有的话就是None

        x = torch.squeeze(x,1)  # 不需要特征这个维度(原因不明)，只能接收3个参数
        r_out, (h_n, h_c) = self.rnn(x,  None)
        out = self.out(r_out[:,-1,:])  # (batch, time_step, output_size)，time_step设为-1，取最后一个时刻的值
        return out


def rnn_train():

    train_data = dsets.EMNIST(
        root='./mnist',
        split='mnist',
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOADS
    )
    train_load = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)




    rnn = RNN()
    optimizer = op.Adam(params=rnn.parameters(),lr=LR)
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step ,(t_x ,t_y) in enumerate(train_load):
            y = rnn(t_x)
            loss = loss_fun(y,t_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("step",step,"| loss=",loss)

    torch.save(rnn,"./rnn.pkl")

def rnn_test():
    test_data = dsets.EMNIST('./mnist', 'mnist', train=False)

    test_x = test_data.test_data.type(torch.FloatTensor) / 255.
    test_y = test_data.test_labels


    rnn = torch.load('./rnn.pkl')
    y = rnn(test_x)
    y = torch.max(y,1)[1]
    count = 0
    for i in range(len(y)):
        if y[i]==test_y[i]:
            count+=1
    print(count/len(y))   # 96.85%   没有cnn强大，因为本来就不是用来做这个的，适合时间序列


if __name__ == '__main__':
    # rnn_train()
    rnn_test()
