
import numpy as np
from numpy import poly1d
from scipy import linalg ,optimize


def main():
    a = np.array([1,2,3])
    t = np.hstack(([[3,2,1],[33,44,55]],[[1,2,3],[11,22,33]]))
    print(t)
    t = np.vstack(([33,44,55],[11,22,33]))
    print(t.shape)
    print(t)
    a = np.array([[1,2,3],[4,5,6]])
    b= np.array([[7,8,9],[11,12,13]])
    t = np.c_[[1,2,3],[4,5,6]]
    t = np.r_[[[1,2,3],[11,22,33]],[[4,5,6],[44,55,66]]]
    t = np.r_[[11,22,33],[44,55,66]]
    print(t.shape)
    print(t)



def f(x):
    print(id(x))


if __name__ == '__main__':
    main()