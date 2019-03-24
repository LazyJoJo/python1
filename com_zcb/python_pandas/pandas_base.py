# pandas 基础

import pandas as pd

def main():

    # series ：如果index没给，会自动添加从0开始,有点像Java数组，只是竖着排列
    a = pd.Series([[11,22],[33,44]],index=[1,2])

    # --------------------------------------------------------------------------

    data = [[2000, 'Ohino', 1.5],
            [2001, 'Ohino', 1.7],
            [2002, 'Ohino', 3.6],
            [2001, 'Nevada', 2.4],
            [2002, 'Nevada', 2.9]]  # type(data) 为 list
    data2 = {"l1":[1,2,3],"l2":[4,5,6]}

    # list to series
    ser = pd.Series(data, index=['one', 'two', 'three', 'four', 'five'])

    # dataframe 就以下三种创建方式，传入series默认是有index，传入dict默认是有columns,已有的就不能重写了，否则数据丢失
    # 第三种，data如果是矩阵，应该是可以用numpy的一些矩阵生成函数替换数据源的（还没测过）
    df = pd.DataFrame(ser,columns=['series'] ) # series 只是dataframe中的一列，df有点像一个矩阵把多个series拼在一起，并且生成一个默认的列名从0开始
    df = pd.DataFrame(data2,index=[1,2,23]) # 传入对象是dict时，key变成column，并且不能重写，value会变成每一列的值
    df = pd.DataFrame(data, columns=[1, 2, 3], index=[1, 2, 3, 4, 5])  # 如果传入数据是二维list 则会自动对应成矩阵

    print(ser)
    print(df)

    print(df.keys())  # 对df而言keys是columns ，
    print(ser.keys())  # 对series而言keys是index

    #  dataframe具体操作

    


if __name__ == '__main__':
    main()