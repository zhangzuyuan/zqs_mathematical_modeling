from typing import AnyStr
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import *

#求积分 I = f（a,b） f(x)dx的近似值
#输入端点a，b；正偶数n；
#输出I的近似值XI；
def f(x):
    res=math.sin(x);
    return res;

def CompositeSimpsonRule(a,b,n):
    h=(b-a)/n;
    XI0=f(a)+f(b);
    XI1=0;#f(x_(2i-1))的和
    XI2=0;#f(x_(2i))的和
    for i in range(n):
        X=a+i*h;
        if i%2==0:
            XI2=XI2+f(X);
        else:
            XI1=XI1+f(X);
    XI=h*(XI0+2*XI2+4*XI1)/3;
    return XI;

if __name__ == "__main__":
    ans=CompositeSimpsonRule(0,math.pi,20);
    print(ans);