
import numpy as np
import matplotlib.pyplot as plt


def mesh2d_(f,*args):
    """ Plot function taking len 2 vector as single argument
          f(xs)
    """
    def g(x,y,*args):
        return f(np.array([x,y]),*args)
    mesh2d(g,*args)


def mesh2d(f,*args):
    """ Plot function taking two arguments
        f(x,y)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y =  np.arange(-0.5, 0.5, 0.005)
    X, Y = np.meshgrid(x, y)
    zs = np.array([ f(x_,y_,*args) for x_,y_ in zip( np.ravel(X), np.ravel(Y)) ])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()