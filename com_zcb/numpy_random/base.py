import numpy as np


def main():
    a = np.random.randint(3,5,[2,3])
    print(type(a))
    a = ["dfd","df",'df','gfgd','qwe']
    b = np.random.choice(a,3,replace=False)
    b = np.random.permutation(a)
    print(a)
    np.random.shuffle(a)
    print(a)
    print(b)

    mylist = [1,"dfd",True]
    x = np.array(mylist)
    print(x)
    np.ones([3,3],float)
    print(np.arange(4))
    print(1/3)



if __name__ == '__main__':
    main()