from timemachines.skatertools.data.long import random_long_data

# Pulls 'fathom' data which has reasonable signal but plenty of noise too

if __name__=='__main__':
    import matplotlib.pyplot as plt
    y,t = random_long_data(n_obs=10000)
    plt.plot(t,y)
    plt.grid()
    plt.show()