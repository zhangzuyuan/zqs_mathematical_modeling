#试位法来加速牛顿法的收敛

import math
#给定区间[p0,p1]上的连续函数f,这里f(p0)和f(p1)反号，求解f(x)=0;
#输入近似解p0,p1，精度要求TOL；最大迭代次数N0
#输出近似解p或者失败信息

def f(x):
    res=math.cos(x)-x;
    return res;

def FalsePosition(p0,p1,TOL,N0):
    q0=f(p0);
    q1=f(p1);
    for i in range(1,N0):
        p=p1-q1*(p1-p0)/(q1-q0);
        if abs(p-p1)<TOL:
            return p;
        q=f(p);
        if q*q1<0:
            p0=p1;
            q0=q1;
        p1=p;
        q1=q;
    print("The method failed after N0 iterations,N0=",N0);



if __name__ == "__main__":
    ans=FalsePosition(0.5,math.pi/4,0.0001,40);
    print(ans);