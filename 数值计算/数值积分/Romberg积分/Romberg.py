from typing import AnyStr
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import *

#求积分 I = f（a,b） f(x)dx的近似值
#输入端点a，b；正偶数n；
#输出数组R
def f(x):
    res=math.sin(x);
    return res;

def Romberg(a,b,n):
    h=(b-a);
    R=np.zeros([n+1,n+1]);
    R[1][1]=(h/2)*(f(a)+f(b));
    for i in range(2,n+1):
        for
        for j in range(2,i+1):
            R[2][j]=R[2][j-1]

    

if __name__ == "__main__":
    ans=Romberg(0,math.pi,20);
    print(ans);