import math
#给定初始近似值p0，求p=g（p）的解
#输入近似解p0，精度要求TOL；最大迭代次数N0
#输出近似解p或者失败信息
def g(x):
    res=x-(x**3+4*x**2-10)/(3*x**2+8*x);
    return res;

def FixedPointIteration(p0,TOL,N0):
    for i in range(N0):
        p=g(p0);
        if abs(p-p0)<TOL:
            return p;
        p0=p;
    print("The method failed after N0 iterations,N0=",N0);

if __name__ == "__main__":
    ans=FixedPointIteration(1.5,0.0001,40);
    print(ans);
