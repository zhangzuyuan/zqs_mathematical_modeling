#当牛顿法的导数很难求得时候用正割法

import math
#给定初始近似值p0,p1，求f(x)=0的解
#输入近似解p0,p1，精度要求TOL；最大迭代次数N0
#输出近似解p或者失败信息

def f(x):
    res=math.cos(x)-x;
    return res;

def secant(p0,p1,TOL,N0):
    q0=f(p0);
    q1=f(p1);
    for i in range(1,N0):
        p=p1-q1*(p1-p0)/(q1-q0);
        if abs(p-p1)<TOL:
            return p;
        p0=p1;
        q0=q1;
        p1=p;
        q1=f(p);
    print("The method failed after N0 iterations,N0=",N0);



if __name__ == "__main__":
    ans=secant(0.5,math.pi/4,0.0001,40);
    print(ans);