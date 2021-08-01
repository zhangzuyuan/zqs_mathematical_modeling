from typing import AnyStr
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, fill_between
import numpy as np
from numpy import *

#求积分 I = f（a,b） f(x)dx的近似值
#输入端点a，b；精度要求TOL，层次数的界N；
#输出I的近似值APP或超过N的信息；
def f(x):
    res=(100/(x**2))*math.sin(10/x);
    return res;

def AdaptiveQuadrature(a,b,TOL,N):
    APP=0;
    i=1;
    TOLi=np.zeros(1+1);TOLi[i]=10*TOL;
    ai=np.zeros(1+1);ai[i]=a;
    h=np.zeros(1+1);h[i]=(b-a)/2;
    FA=np.zeros(1+1);FA[i]=f(a);
    FC=np.zeros(1+1);FC[i]=f(a+h[i]);
    FB=np.zeros(1+1);FB[i]=f(b);
    S=np.zeros(1+1);S[i]=h[i]*(FA[i]+4*FC[i]+FB[i])/3;
    L=np.zeros(1+1);L[i]=1;

    while i>0:
        FD=f(ai[i]+h[i]/2);
        FE=f(ai[i]+3*h[i]/2);
        S1=h[i]*(FA[i]+4*FD+FC[i])/6;
        S2=h[i]*(FC[i]+4*FE+FB[i])/6;
        v1=ai[i];
        v2=FA[i];
        v3=FC[i];
        v4=FB[i];
        v5=h[i];
        v6=TOLi[i];
        v7=S[i];
        v8=L[i];
        i=i-1;
        if abs(S1+S2-v7)<v6:
            APP=APP+(S1+S2);
        else:
            if v8>=N:
                print("LEVEL EXCEEDED");
                return 0;
            else:
                i=i+1;
                #右半区间
                if i>=len(L):
                    TOLi=np.append(TOLi,0);
                    ai=np.append(ai,0);
                    h=np.append(h,0);
                    FA=np.append(FA,0);
                    FC=np.append(FC,0);
                    FB=np.append(FB,0);
                    S=np.append(S,0);
                    L=np.append(L,0);
                ai[i]=v1+v5;
                FA[i]=v3;
                FC[i]=FE;
                FB[i]=v4;
                h[i]=v5/2;
                TOLi[i]=v6/2;
                S[i]=S2;
                L[i]=v8+1;
                i=i+1;
                #左半区间
                if i>=len(L):
                    TOLi=np.append(TOLi,0);
                    ai=np.append(ai,0);
                    h=np.append(h,0);
                    FA=np.append(FA,0);
                    FC=np.append(FC,0);
                    FB=np.append(FB,0);
                    S=np.append(S,0);
                    L=np.append(L,0);
                ai[i]=v1;
                FA[i]=v2;
                FC[i]=FD;
                FB[i]=v3;
                h[i]=h[i-1];
                TOLi[i]=TOLi[i-1];
                S[i]=S1;
                L[i]=L[i-1];
    

    return APP;

if __name__ == "__main__":
    ans=AdaptiveQuadrature(1,3,0.0001,100);
    print(ans);