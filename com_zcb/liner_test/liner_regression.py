import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.array([
        [1,6],
        [2,5],
        [3,7],
        [4,10]
    ])
    m = len(data)
    x = np.array([np.ones(m), data[:,0]]).T
    print(data[:,0])
    print(x)

if __name__ == '__main__':
    main()