
import numpy as np
from numpy import poly1d
from scipy import linalg ,optimize


def main():
    a = np.array([0,1,2])
    t = np.tile(a,())
    print(t.shape)
    print(t)



def f(x):
    print(id(x))


if __name__ == '__main__':
    main()