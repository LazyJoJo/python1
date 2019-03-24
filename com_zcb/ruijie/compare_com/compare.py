
# 用以比较组件的准确性
#
#

import numpy as np
import pandas as pd
import re
def com_count(path):
    df = pd.read_table(path,header=-1)
    df = df.loc[:,0]
    ans = set()
    for i in range(len(df)):
        sta = df[i]
        if sta.find(',')!=-1:
            sta = sta.split(' ')
            for i in range(len(sta)):
                s = sta[i]
                if s.endswith(','):
                    s = s[0:-1]
                    # print(s)
                    ans.add(s)
        else:
            sta = sta.split(" ")
            if len(sta)>=2:
                sta = sta[1]
            else:
                sta = sta[0]
            s = re.match('(\S+)_\d+',sta)

            if s==None:
                s = sta
            else:
                s = s.group(1)  # 获取匹配数组

            ans.add(s)
    return ans

def read_xls(path):
    df = pd.read_excel(path,header=0)
    df = df.loc[:,'组件名称']
    return set(df)

if __name__ == '__main__':

    s1 = com_count('~/Desktop/comp/comp1.txt')
    s2 = com_count('~/Desktop/comp/comp2.txt')
    s3 = com_count('~/Desktop/comp/comp3.txt')
    sall = s1.union(s2).union(s3)

    ss = read_xls("~/Desktop/com.xlsx")
    print(len(ss))
    print(len(sall))
    lose = sall.difference(ss)
    error = ss.difference(sall)
    for i in lose:
        print(i+'\r')
    print('=================')
    for i in error:
        print(i+'\r')

    d = {"cmdd中丢失数据":pd.Series(list(lose))}
    lose_df = pd.DataFrame(d)
    d = { "基线中不存在，但cmdd中有的数据":pd.Series(list(error))}
    error_df = pd.DataFrame(d)
    print(error_df)
    ans = pd.concat([lose_df,error_df],axis=1)
    print(ans)
    ans.to_csv('~/Desktop/ans.csv', header=True, index=False, encoding='utf_8_sig')



    # pd.excel_w



