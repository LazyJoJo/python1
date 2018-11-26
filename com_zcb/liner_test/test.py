import numpy as np
import matplotlib.pyplot as plt
def main():

    # np.random.seed(7234)
    a = np.random.randn(2,1)
    b = np.random.randn(2,1).reshape(-1,1)
    print(a-b)





if __name__ == '__main__':
    main()