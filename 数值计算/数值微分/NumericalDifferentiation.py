from typing import AnyStr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import *

#三点公式
#向前/后差 公式
def ThreePointEndPointFormula(n,a,b,fx):
    h=(b-a)/n;
    dfx=np.zeros(n+1);
    for i in range(n):
        dfx[i]=(1/2*h[i])*();
    
    return dfx;

if __name__ == "__main__":
    x,a=get_data();
    ans=NaturalCubicSpline(20,x,a);
    draw(x,ans);
    print(ans);