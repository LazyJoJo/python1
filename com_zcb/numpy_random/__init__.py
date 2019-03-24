import numpy as np
import matplotlib.pyplot as plt

def main():
    a = np.random.uniform(0,10,100)
    # print(a)
    # print(np.mean(a))
    count, bins, ignored = plt.hist(a,15,density=True)
    print(count)
    print(bins)
    # print(ignored)
    plt.show()

if __name__ == '__main__':
    main()