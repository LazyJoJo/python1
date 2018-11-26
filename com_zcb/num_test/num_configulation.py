
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

def read_csv(filename):
    all = pd.read_csv(filename)
    # print(type(all['label'])) # Series 传入参数不同解析的结果也不同，只能传入单个字符串，如果多个，就要传入一个list
    # print(all.columns.values) # 知道如何获取column就好 index类似（注意没有s)
    label = all['label'] # 这里是series ，要得到DataFrame 传入list all[['label']]
    label = label.astype('int')
    feature = all.iloc[:,1:] # 分割dataFrame用新的方式（类似numpy）
    # print(type(all.index.values))  # ndarray
    # print(feature)
    # feature[feature>0]=1 # 不知道是否有替代方案
    feature = np.array(feature)
    label = np.array(label)

    return feature, label



def read_test_csv(file):

    a = pd.read_csv(file)
    a = a.astype('int')
    a = np.array(a)
    return a

def main():
    train_data,train_tag = read_csv('/users/zcb/desktop/num_test/train.csv')
    # test_data = read_test_csv('/users/zcb/desktop/num_test/test.csv')
    # print(train_data.shape)
    train = train_data[:30000,:]
    test = train_data[-500:,:]
    tag = train_tag[:30000]
    right = train_tag[-500:] # 正确答案
    # ans = my_knn(train,tag,test)
    ans = skl_knn(train,tag,test)
    df = pd.DataFrame({"ImageId": np.arange(len(ans)) + 1, "Label": ans})
    df.to_csv("/users/zcb/desktop/num_test/test_ans.csv",index=False)
    df = pd.DataFrame({"ImageId": np.arange(len(right)) + 1, "Label": right})
    df.to_csv('/users/zcb/desktop/num_test/right_ans.csv',index=False)
    # print(ans)

def skl_knn(train_feature ,train_label, test_feature, k=5):
    neigh = KNeighborsClassifier(algorithm='auto',weights='distance')
    neigh.fit(train_feature ,train_label)
    ans = neigh.predict(test_feature)
    print(ans)
    return ans


def my_knn(train_feature ,train_label, test_feature, k=5 ):
    num = train_feature.shape[0]
    size = test_feature.shape[0]
    test_label = []
    for i in range(size):
        m = test_feature[i,:]
        mcopy = np.tile(m,(num,1))
        mf = mcopy-train_feature
        mf = mf*mf
        mf = mf.sum(axis=1)
        mf = mf**(1/2.0)
        m = np.argsort(mf) # 应该是由小到大，距离越小的越像
        # print(mf[m[0]])
        class_type = {}
        sum = 0
        for j in range(k):
            sum = sum+mf[m[j]]
        for j in range(k):
            label = train_label[m[j]]
            class_type[label] = class_type.get(label,0)+sum/mf[m[j]]  #get设置默认值
        print(class_type)

        ans = sorted(class_type.items(),key=lambda x:x[1],reverse=True)
        ans = ans[0][0]
        print(i+1,ans)
        print("-----------------")
        # print(ans)
        test_label.append(ans)
    return test_label

def compare_file(file1,file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    # print(df1)
    label1 = df1["Label"]
    label2 = df2["Label"]
    count = 0
    for i in range(len(label1)):
        if label1[i]==label2[i]:
            count+=1
        else:
            print(i+1,label1[i],label2[i])
    print(float(count)/float(len(label1)))
def test():
    # train_data, train_tag = read_csv('/users/zcb/desktop/num_test/train.csv')
    # print(type(train_tag))
    # df = pd.DataFrame( {"ImageId":np.arange(len(train_tag))+1,"Label":train_tag})
    # df.to_csv("/users/zcb/desktop/num_test/test_ans.csv",index=False)
    l = []
    # with open("/users/zcb/desktop/value","r") as file:
    #     for line in file:
    #         line = line.strip("\n")
    #         l.append(line)
    # l = map(int,l)
    # for i in l:
    #
    #     print(l )
    df = pd.DataFrame({"ImageId": np.arange(len(l)) + 1, "Label": l})
    df.to_csv("/users/zcb/desktop/num_test/test_ans.csv", index=False)



if __name__ == '__main__':
    main()
    # test()
    compare_file("/users/zcb/desktop/num_test/test_ans.csv","/users/zcb/desktop/num_test/right_ans.csv")