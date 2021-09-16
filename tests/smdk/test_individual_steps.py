from timemachines.skaters.smdk.smdkinclusion import using_simdkalman, using_latest_simdkalman

if using_simdkalman:
    from timemachines.skaters.smdk.smdkexample import smdk_example
    from simdkalman import KalmanFilter

    def test_evolve_same_A():
        A, H, m0, P0, Q, R,y0 = smdk_example(different_transitions=False)
        kf = KalmanFilter(
            state_transition=A,   # A
            process_noise=Q,      # Q
            observation_model=H,  # H
            observation_noise=R)  # R
        results = kf.predict(data=y0,n_test=1,initial_value=m0, initial_covariance=P0,
                             states=True, covariances=True)
        m1 = results.states.mean
        P1 = results.states.cov
        if False:
            assert m1.shape == m0.shape
            assert P1.shape == P0.shape

    def test_update_same_A():
        A, H, m0, P0, Q, R, y0 = smdk_example(different_transitions=False)
        kf = KalmanFilter(
            state_transition=A,  # A
            process_noise=Q,  # Q
            observation_model=H,  # H
            observation_noise=R)  # R
        m1, P1, K, ll = kf.update(m0, P0, y0, log_likelihood=True)
        assert m1.shape == m0.shape
        assert P1.shape == P0.shape


    if using_latest_simdkalman:

        def test_evolve_different_A():
            A, H, m0, P0, Q, R, y0 = smdk_example(different_transitions=True)
            kf = KalmanFilter(
                state_transition=A,  # A
                process_noise=Q,  # Q
                observation_model=H,  # H
                observation_noise=R)  # R
            results = kf.predict(data=y0, n_test=1, initial_value=m0, initial_covariance=P0,
                                 states=True, covariances=True)
            m1 = results.states.mean
            P1 = results.states.cov

            if False:
                # Can't do this...because m1 is a different shape
                results2 = kf.predict(data=y0, n_test=1, initial_value=m1, initial_covariance=P1,
                                     states=True, covariances=True)
                m2 = results.states.mean
                P2 = results.states.cov

            if False:
                assert m1.shape == m0.shape
                assert P1.shape == P0.shape

        def test_update_different_A():
            A, H, m0, P0, Q, R, y0 = smdk_example(different_transitions=True)
            kf = KalmanFilter(
                state_transition=A,  # A
                process_noise=Q,  # Q
                observation_model=H,  # H
                observation_noise=R)  # R
            m1, P1, K, ll = kf.update(m0, P0, y0, log_likelihood=True)
            assert m1.shape==m0.shape
            assert P1.shape==P0.shape


if __name__=='__main__':
    assert using_latest_simdkalman, '  # pip install --upgrade git+https://github.com/oseiskar/simdkalman should fix'
    test_update_same_A()
    test_evolve_same_A()
    test_update_different_A()
    test_evolve_different_A()