from timemachines.synthetic import brownian_with_exogenous


if __name__=='__main__':
    import matplotlib.pyplot as plt
    y = brownian_with_exogenous(n=50)
    plt.plot(y)
    plt.legend(['target','exogenous'])
    plt.grid()
    plt.show()