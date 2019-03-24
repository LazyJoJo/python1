
#基础Python使用记录
import matplotlib.pyplot as plt
import json

def main():
    a = [1,2] # list
    b = (3,4) # tuple
    c = {"1":"3","2":"22"} # dict

    #dict set zip的这类写法与强转的写法是一致的
    c = dict(t1=[11,33],t2=[22,33])  #key不能是数字
    c = dict(one=[1, 1,],two = [ 2,  2]) #这种写法恨适合给别的方法传参数，参数名都是不用加冒号的，看起来类似
    #plt.hist(b,**c)  #传参的用法是这样的

    d = set(c) #传a,b也可以
    d = {1,2,3,4} # 这种方式也是set，但是必须内部有值，所以看到{}不一定是dict也有可能是set
    c = {} # 这个是初始化dict，不是set，set初始化只能是set()有点像强转的意思，可能后台实现就是强转
    e = zip(b,b) # 传list或tuple  结果是一个tuple的List
    print(c)


    print([num for num in range(0,10) if num%2==0]) #最前面的是返回值，后面的for if 表达式

    x = '{"name":"john","age":30,"city":"new york"}'
    y = json.loads(x)  #反序列化，将str 变成 Python object
    print(y["age"])
    print(json.dumps(y)) #序列化，将object变成str

    mytuple = ("apple",'banana','cherry')
    myit = iter(mytuple)  #变成迭代形式
    print(next(myit))



if __name__ == '__main__':
    main()