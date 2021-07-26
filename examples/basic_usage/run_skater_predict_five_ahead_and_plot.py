from timemachines.skaters.simple.thinking import thinking_slow_and_slow
from timemachines.skatertools.visualization.priorplot import prior_plot
from timemachines.skatertools.data.real import hospital
import matplotlib.pyplot as plt

if __name__=='__main__':
    y = hospital(n=200)
    prior_plot(f=thinking_slow_and_slow,y=y,k=5)
    plt.show()