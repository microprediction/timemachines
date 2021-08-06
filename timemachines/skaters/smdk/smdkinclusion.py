try:
    import simdkalman
    using_simdkalman = True
except ImportError:
    using_simdkalman = False
    using_latest_simdkalman = False


if using_simdkalman:
    # Also checks whether we can use the version supporting multiple systems
    # pip install --upgrade git+https://github.com/oseiskar/simdkalman should fix
    from timemachines.skaters.smdk.smdkexample import smdk_example
    A, H, m0, P0, Q, R, y0 = smdk_example(different_transitions=False)
    from simdkalman import KalmanFilter
    kf = KalmanFilter(state_transition=A, process_noise=Q, observation_model=H,observation_noise=R)
    try:
        A, H, m0, P0, Q, R, y0 = smdk_example(different_transitions=True)
        kf = KalmanFilter(state_transition=A, process_noise=Q, observation_model=H, observation_noise=R)
        using_latest_simdkalman = True
    except AssertionError:
        using_latest_simdkalman = False
