import numpy as np
import pandas as pd

def main():
    print(15/17*(0.6*np.log2(0.6)+0.4*np.log2(0.4))+0.998)
    print(12/17*np.log2(12/17)+5/17*np.log2(5/17))

if __name__ == '__main__':
    main()