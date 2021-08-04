import numpy as np


def smdk_example(different_transitions=False):
    """ Example multi-dimensional Kalman filter system parameters

    :param different_transitions:
    :return:   A,H,m0,P0,Q,R,y0
    """

    # Process noise
    Q = np.zeros((2 ,3, 3))
    Q[0, :, :] = np.eye(3)
    Q[1, :, :] = np.eye(3)

    # Prior state means
    should_we_transpose = True
    if should_we_transpose:
        m0 = np.zeros((2, 3, 1))
        m0[0, :, :] = np.array([[1.5, 1.5, 1.5]]).transpose()
        m0[1, :, :] = np.array([[1.5, 1.5, 1.5]]).transpose()
    else:
        m0 = np.zeros((2, 1, 3))
        m0[0, :, :] = np.array([[1.5, 1.5, 1.5]])
        m0[1, :, :] = np.array([[1.5, 1.5, 1.5]])

    # Prior covariances
    P0 = np.zeros((2, 3, 3))
    P0[0, :, :] = np.eye(3)
    P0[1, :, :] = np.eye(3)

    # Observations
    y0 = np.zeros((2, 1, 1))
    y0[0, :, :] = np.array([[-1.4]])
    y0[1, :, :] = np.array([[-1.4]])

    # Measurement var
    R = np.zeros((2, 1, 1))
    R[0, :, :] = np.eye(1)
    R[1, :, :] = 0.6 * np.eye(1)

    # Observation equations
    H0 = np.array([[0.8, 0.2, 0]])
    H1 = np.array([[0.8, 0.2, 0]])
    H0.shape = (1, 3)
    H1.shape = (1, 3)
    H = np.zeros((2, 1, 3))
    H[0, :, :] = H0
    H[1, :, :] = H1

    # Single observation
    y0 = np.zeros((2, 1, 1)) # First dim indexes the time-series, second the time-step, third the observation dimension
    y0[0, :, :] = np.array([[-1.4]])
    y0[1, :, :] = np.array([[-1.7]])

    # Transition(s)
    A0 = np.array([[0.5, 0.3, 0.2], [1, 0, 0], [0, 1, 0]])
    if different_transitions:
        A = np.zeros((2, 3, 3))
        A1 = np.array([[0.4, 0.4, 0.2], [1, 0, 0], [0, 1, 0]])
        A[0, :, :] = A0
        A[1, :, :] = A1
    else:
        A = A0

    return A, H, m0, P0, Q, R, y0
