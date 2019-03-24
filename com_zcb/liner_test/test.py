import numpy as np
import matplotlib.pyplot as plt
import random
import math
def main():

    # np.random.seed(7234)
    # a = np.random.randn(2,1)
    # b = np.random.randn(2,1).reshape(-1,1)
    # print(a)
    # print(b)
    # a = np.random.random(10,1,30)
    # print(a)
    # print(a-b)
    ans = []
    ans_1 = []
    ans_2 = []

    for i in range(100000):
        a = list(range(1,31))
        random.shuffle(a)
        # print(a)
        b = 0
        num = 12
        m = max(a[0:num])
        for j in range(29-num):
            if m<a[num+j]:
                b = a[num+j]
        if b==0:
            b=a[-1]

        ans.append(b)
    print(compute(ans))

def compute(l):
    count = 0
    for i in range(len(l)):
        if l[i]==30:
            count+=1
    return count/len(l)






if __name__ == '__main__':
    main()