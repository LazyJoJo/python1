import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre

def main():
    # data = [list("adLi"),list("dsfL"),list("dsLL")]
    # dd = pd.Series(['1','2','a','s','b'])
    # datas = pd.DataFrame(data,columns=['Lqq','Lww','ee','rr'],index=["L","d","L"])
    # a = np.arange(9)
    # b = np.arange(3,12)
    # a = a.reshape(3,3)
    # b = b.reshape(3,3)
    # datas = pd.DataFrame(b,columns=list("asd"))
    # print(zip(a,b))
    # for i in zip(a,b):
    #     print(i)
    #
    # # print(datas.loc['L',['Lqq','Lww']])
    # print(datas.groupby(['a','s']).mean())
    # print(pd.date_range('2018-04-21','2018-04-24',periods=4))
    X,y = make_blobs(n_samples=1000 , centers=20,random_state=123)
    print(X)
    print(y)
    labels = ['r','b']
    y = np.take(labels ,(y<10))
    mask = y=='r'
    print(mask)
    print(type(X))
    print(type(X[mask,0]))
    print(X[:5,0])
    print(X[:5][0])
    for label in labels:
        mask = (y == label)
        plt.scatter(X[mask, 0], X[mask, 1], c=label)
    plt.show()
    # print(X[mask,0],X[mask,1])
    df = pd.DataFrame(y) # 10
    print(df.describe())  # 0
    df2= pd.DataFrame(X)
    # print(df2)
    print(df2[1].describe())
    print(df.head())
    # df["color"]=np.take(['r',"d",'c'],(df[0]>10))  #false 0, true 1
    # print(df)

    scaler = pre.StandardScaler()
    scaler_df = scaler.fit_transform(df)
    





if __name__ == '__main__':
    main()
