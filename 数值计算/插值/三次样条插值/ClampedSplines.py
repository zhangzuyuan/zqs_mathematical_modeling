
from typing import AnyStr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import *

#对于函数f，构造定义在点x0<x1<...<xn,上满足S'(x0)=f'(x0)和S'(xn)=f'(xn)的三次样条插值S；
#输入n;x0,x1,...,xn;a0=f(x0),a1=f(x1),...,an=f(xn);FPO=f'(x0);FPN=f'(xn);
#输出a_j,b_j,c_j,d_j(j=0,1,...,n-1)
#S(x)=S_j(x)=a_j+b_j*(x-x_j)+c_j*(x-x_j)**2+d_j*(x-x_j)**3;(x_j<=x<=x_j+1)
def NaturalCubicSpline(n,x,a,FPO,FPN):
    h=np.zeros(n);
    ans=np.zeros([n,4]);
    alpha=np.zeros([n,1]);
    l=np.zeros(n+1);
    u=np.zeros(n+1);
    z=np.zeros(n+1);
    c=np.zeros(n+1);
    b=np.zeros(n+1);
    d=np.zeros(n+1);
    for i in range(n):
        h[i]=x[i+1]-x[i];
    for i in range(1,n):
        alpha[i]=( (3/h[i])*(a[i+1]-a[i]) )-( (3/h[i-1])*(a[i]-a[i-1]) );
    l[0]=1;
    u[0]=0;
    z[0]=0;
    for i in range(1,n):
        l[i]=2*(x[i+1]-x[i-1])-h[i-1]*u[i-1];
        u[i]=h[i]/l[i];
        z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i];
    l[n]=1;
    z[n]=0;
    c[n]=0;
    for j in range(n-1,-1,-1):
        c[j]=z[j]-u[j]*c[j+1];
        b[j]=(a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3;
        d[j]=(c[j+1]-c[j])/(3*h[j]);
    for i in range(n):
        ans[i][0]=a[i];
        ans[i][1]=b[i];
        ans[i][2]=c[i];
        ans[i][3]=d[i];
    return ans;

def get_data():
    x=np.array([0.9,1.3,1.9,2.1,2.6,3.0,3.9,4.4,4.7,5.0,6.0,7.0,8.0,9.2,10.5,11.3,11.6,12.0,12.6,13.0,13.3],dtype='float');
    a=np.array([1.3,1.5,1.85,2.1,2.6,2.7,2.4,2.15,2.05,2.1,2.25,2.3,2.25,1.95,1.4,0.9,0.7,0.6,0.5,0.4,0.25],dtype='float');
    return x,a;

def draw(X,ans):
    plt.figure(figsize=(15,4))
    i=0;
    for tmp in ans:
        x = np.linspace(X[i],X[i+1], 100)
        y = tmp[0]+tmp[1]*(x-X[i])+tmp[2]*(x-X[i])**2+tmp[3]*(x-X[i])**3;
        plt.plot(x, y, color='b', linestyle='-')
        i=i+1;
    plt.legend(['threeNaturalSplines'])
    x_major_locator=MultipleLocator(1)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(1)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0,14)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0,4)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.grid(True) ##增加格点
    plt.show()

if __name__ == "__main__":
    x,a=get_data();
    ans=NaturalCubicSpline(20,x,a);
    draw(x,ans);
    print(ans);