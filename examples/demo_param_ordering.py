
import matplotlib.pyplot as plt
import numpy as np
from timemachines.conventions import to_log_space


# Illustrates mapping from (0,1) to parameters in R^3




if __name__=='__main__':

    bounds = [(-300,300), (-1,1), (-3,3) ]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    step = 0.0005

    vs = list()
    ts = list()
    tail_len = 1000

    rs = np.linspace(start=0, stop=1, num=1000)
    for step_no, r in enumerate(rs):
        ts.append(r)
        v = to_log_space(r=r, bounds=bounds)
        vs.append(v)
        trailing = vs[-tail_len:]
        tail = max(ts[:-tail_len] + [0.0001])
        x, y, z = zip(*trailing)
        ax.clear()
        ax.plot(x, y, z)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=100)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel('Most important parameter')
        ax.set_ylabel('Second most important')
        ax.set_zlabel('Third most important')
        ax.set_title('t -> (' + str(np.round(tail, 4)).zfill(4) + ',' + str(np.round(r, 4)).zfill(4) + ')')
        plt.show(block=False)
        plt.pause(0.05)
        if step_no == 0:
            plt.pause(2)

    x, y, z = zip(*vs)
    ax.plot(x, y, z)
    plt.show()

    plt.title('Parameter convention (1-dim)')
    plt.grid()
    plt.show()