import numpy as np
import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import eli5
from eli5.sklearn import PermutationImportance

def main():
    df = pd.read_csv('/users/zcb/desktop/num_test/train.csv')
    label = df.columns
    print(label)
    df_target = df.iloc[:,0]
    df_feature = df.iloc[:,1:]

    x_train,x_test,y_train,y_test = train_test_split(df_feature,df_target,train_size=0.7,random_state=0)
    my_model = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
    perm = PermutationImportance(my_model,random_state=1).fit(x_test,y_test)
    df = eli5.explain_weights_df(perm)
    print(df)




if __name__ == '__main__':
    main()









