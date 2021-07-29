
#f 是区间[a,b]上的连续函数，且f(a)和f(b)反号，求f(x)=0的解的方法：
#输入端点a，b，精度要求TOL；最大迭代次数N0
#输出近似解p或者失败信息
def f(x):
    res= x**3+4*x**2-10;
    return res;

def BisectionMethod(a,b,TOL,N0):
    FA=f(a);
    for i in range(N0):
        p=a+(b-a)/2;
        FP=f(p);
        if FP==0 or ((b-a)/2)<TOL:
            return p;
        if FA*FP>0:
            a=p;
            FA=FP;
        else:
            b=p;
    print("Method faild after N0 iterations,N0=",N0);
    return 0;

if __name__ == "__main__":
    ans=BisectionMethod(1,2,0.0001,20);
    print(ans);
