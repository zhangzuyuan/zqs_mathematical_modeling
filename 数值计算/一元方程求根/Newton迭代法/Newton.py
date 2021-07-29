import math
#给定初始近似值p0，求f(x)=0的解
#输入近似解p0，精度要求TOL；最大迭代次数N0
#输出近似解p或者失败信息

def f(x):
    res=math.cos(x)-x;
    return res;
def df(x):
    res=-math.sin(x)-1;
    return res;

def newton(p0,TOL,N0):
    for i in range(N0):
        p=p0-f(p0)/df(p0);
        if abs(p-p0)<TOL:
            return p;
        p0=p;
    print("The method failed after N0 iterations,N0=",N0);



if __name__ == "__main__":
    ans=newton(math.pi/4,0.0001,40);
    print(ans);