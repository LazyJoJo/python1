import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def main():
    num = np.random.randint(1,7,10000)
    num_size = 50
    print(np.std(num))
    print(np.mean(num))
    meanList = []
    for i in range(1000):
        d = []
        for j in range(num_size):
            d.append(num[int(np.random.random()*len(num))])


        meanList.append(np.mean(d))
    print(np.std(meanList))
    print(1.7/pow(1000,0.5))
    print(pow(1000,0.5))
    # kwarge = dict(alpha=0.3,normed=True,bins=40)
    plt.hist(meanList)
    # plt.show()



if __name__ == '__main__':
    main()