import sklearn.preprocessing as pre
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def main():
    np.random.seed(1)
    df = pd.DataFrame({
        'x1': np.random.normal(0,2,10000),
        'x2': np.random.normal(5,3,10000),
        'x3': np.random.normal(-5,5,10000)
    })
    scaler = pre.StandardScaler()
    scaler_df = scaler.fit_transform(df)  # np.ndarray
    scaler_df = pd.DataFrame(scaler_df,columns=['x1','x2','x3'])

    my_scaler_df = df.copy()
    my_scaler_df.x1 = df.x1.map(lambda x:(x-0)/2.0)
    my_scaler_df.x2 = df.x2.map(lambda x:(x-5)/3.0)
    my_scaler_df.x3 = df.x3.map(lambda x:(x+5)/5.0)
    print(my_scaler_df.head())
    print(scaler_df.head())
    print(df.head())

    # flg ,(ax1,ax2,ax3) = plt.subplots(ncols=3)
    # sns.kdeplot(df.x2,ax=ax1)
    # sns.kdeplot(scaler_df.x2,ax=ax2)
    # sns.kdeplot(my_scaler_df.x2,ax=ax3)
    # plt.show()









if __name__ == '__main__':
    main()