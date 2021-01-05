
import matplotlib.pyplot as plt
import numpy as np
from timemachines.conventions import to_log_space_1d

if __name__=='__main__':
    us = np.linspace(start=0,stop=1,num=150)
    ys = [to_log_space_1d(u, low=-300, high=700) for u in us]
    plt.plot(us, ys)
    plt.title('Parameter convention (1-dim)')
    plt.grid()
    plt.show()