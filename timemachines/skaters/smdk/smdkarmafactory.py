from timemachines.skaters.smdk.smdkinclusion import using_simdkalman
if using_simdkalman:

    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
    from timemachines.skatertools.components.parade import parade
    from timemachines.skatertools.utilities.nonemath import nonecast
    import numpy as np
    import random
    import math
    from timemachines.skatertools.ensembling.precisionweightedskater import precision_weighted_skater
    from simdkalman.primitives import ddot, ddot_t_right, dinv

    # Pronounced "sim-d-k ARMA factory", not "simd-karma-factory"

    # Not ready for prime time by any means


    def random_p(max_p: int):
        return random.choice(list(range(1, max_p + 1)))


    def random_q(max_q: int):
        return random.choice(list(range(1, max_q + 1)))


    def phi_decay(j):
        return 0.5 ** (j + 1)


    def theta_decay(j):
        return 0.5 ** (j + 1)


    def random_phi(p: int):
        return [(np.random.rand() - 0.5) * phi_decay(j) for j in range(p)]


    def random_theta(q: int):
        return [(np.random.rand() - 0.5) * theta_decay(j) for j in range(q)]


    def smdk_arma_factory(y: Y_TYPE, n_agents: int, max_p: int, max_q:int, s, k: int, a: A_TYPE = None, t: T_TYPE = None,
                          e: E_TYPE = None, r: R_TYPE = None, min_vintage=50):
        """

              max_p - Maximum AR order
              max_q - Maximum MA order

        """
        n_states = max(max_p, max_q+1)
        n_obs = 1  # May generalize later
        assert n_states >= 2
        y0 = wrap(y)[0]
        if not s.get('n_states'):
            s = _arma_initial_state(n_states=n_states, n_agents=n_agents, k=k, x0=y0)
        else:
            assert n_agents == s['n_agents']
            assert n_states == s['n_states']

        if y0 is None:
            return None, s, None
        else:
            # Mutation step - flag those needed system updates
            # Changes s['phi'], s['theta'], s['noise'], s['sigma'] and flags with s['stale']
            if s['n_measurements'] > 10:
                s = _arma_evolution(s,min_vintage=min_vintage)

            # Update system equations if params have mutated
            s = _arma_matrix_update(s, k)

            # Get time step, or default to 1 second
            if t is None:
                dt = 1
            else:
                if s.get('prev_t') is None:
                    s['prev_t'] = -1
                dt = t - s['prev_t']
                s['prev_t'] = t

            # Kalman updates
            prior_mean = ddot(s['A'], s['m'])  # A * m
            prior_cov = ddot(s['A'], ddot_t_right(s['P'], s['A'])) + dt * s['Q']  # A * P * A.t + Q
            measurement = np.array([[[y0]]])
            posterior_mean, posterior_cov, K, ll = _update(prior_mean=prior_mean, prior_covariance=prior_cov,
                                                           observation_model=s['H'], observation_noise=s['R'],
                                                           measurement=measurement,
                                                           log_likelihood=True)
            s['m'] = posterior_mean
            s['P'] = posterior_cov

            # Compute k-step ahead predictions for each agent
            agent_xs = [[np.nan for _ in range(k)] for _ in range(n_agents)]
            agent_stds = [[np.nan for _ in range(k)] for _ in range(n_agents)]
            for j in range(k):
                if j == 0:
                    j_posterior_mean = posterior_mean
                else:
                    j_posterior_mean = ddot(s['powers_of_A'][j - 1, :, :, :], posterior_mean)
                j_y_hat = ddot(s['H'], j_posterior_mean)
                for ndx in range(n_agents):
                    agent_xs[ndx][j] = j_y_hat[ndx, 0, 0]

            # Update agent prediction parades and get their empirical standard errors
            for ndx in range(n_agents):
                _discard_bias, agent_std_j, s['parades'][ndx] = parade(p=s['parades'][ndx], x=agent_xs[ndx], y=y0)
                agent_stds[ndx] = nonecast(agent_std_j, fill_value=1.0)

            # Create the exogenous vector that the precision weighted skater expects.
            # (i.e. y_for_pws[1:] has agent predictions interlaced with their empirical means)
            y_for_pws = [y0]
            s['fitness'] = []
            for agent_x, agent_std in zip(agent_xs, agent_stds):
                y_for_pws.append(agent_x[-1])
                y_for_pws.append(agent_std[-1])
                s['fitness'].append(1./(1e-6+agent_std[-1]**2))

            # Call the precision weighted skater
            x, x_std, s['s_pks'] = precision_weighted_skater(y=y_for_pws, s=s['s_pws'], k=k, a=a, t=t, e=e)
            x_std_fallback = nonecast(x_std, fill_value=1.0)

            s['n_measurements'] += 1
            if s['n_measurements'] < 10:
                # Cold ... just puke naive forecasts
                return [y0] * k, [1.0] * k, s
            else:
                return x, x_std_fallback, s



    def _arma_initial_state(n_states, n_agents, k, x0:float):
        """
        :param n_states:    number of latest states is equal to max(p,q+1)
        :param n_agents:    number of independent ARMA models computed at once
        :param k:           number of steps ahead to forecast
        :param x0:          initial value for all latent lag states
        :return:
        """
        n_obs = 1 # dimension of observation (fixed at 1 for now)
        s = {'n_states': n_states,
             'n_agents': n_agents,
             'n_measurements': 0,
             'm': np.zeros((n_agents, n_states,1)),
             'P': np.zeros((n_agents, n_states, n_states)),
             'H': np.zeros((n_agents, n_obs, n_states)),
             'R': np.zeros((n_agents, n_obs, n_obs)),
             'Q': np.zeros((n_agents, n_states, n_states)),
             'A': np.zeros((n_agents, n_states, n_states)),
             'fitness': [1. for _ in range(n_agents)],
             'prev_t': None,
             's_pws': {}  # State for the precision weighted skater
             }
        ps = [random_p(n_states) for _ in range(n_agents)]
        qs = [random_q(n_states - 1) for _ in range(n_agents)]
        s['phi'] = [random_phi(p) for p in ps]
        s['theta'] = [random_theta(q) for q in qs]
        s['parades'] = [{} for _ in range(n_agents)]  # Track empirical errors individually (somewhat inefficient)
        s['stale'] = [True for _ in range(n_agents)]
        s['r_var'] = [np.random.exponential() ** 4 for _ in range(n_agents)]
        s['q_var'] = [np.random.exponential() ** 4 for _ in range(n_agents)]
        s['powers_of_A'] = np.ndarray((k, n_agents, n_states, n_states))
        s['vintage'] = [ 0 for _ in range(n_agents )]
        # Initialize random states
        for ndx in range(n_agents):
            s['m'][ndx, :, 0] = [x0 for _ in range(n_states)]
            s['P'][ndx, :, :] = (np.random.exponential() ** 4) * np.eye(n_states)
        return s

    def _arma_evolution(s, min_vintage):
        """
            Modify the ARMA population based on fitness
        """
        s['vintage'] = [v + 1 for v in s['vintage']]
        experienced = [ j for j, v in enumerate(s['vintage']) if v>=min_vintage ]
        if any(experienced):
           experienced_fitness = [f for f,v in zip(s['fitness'],s['vintage']) if v >= min_vintage]
           good_fitness = np.percentile(experienced_fitness,q=80)
           poor_fitness = np.percentile(experienced_fitness, q=20)
           good = [ j for j in experienced if s['fitness'][j]>=good_fitness ]
           poor = [ j for j in experienced if s['fitness'][j] <= poor_fitness ]
           n_change = int(math.ceil(s['n_agents']/100))
           for _ in range(n_change):
               a,b,c = list(np.random.choice(good,size=3,replace=False))
               d = np.random.choice(poor)
               s['stale'][d]=True
               s['vintage'][d]=0
               evol_type = random.choice(['ar','ma','r_var','q_var'])
               if evol_type=='ma':
                   s['theta'][d] = [ theta_a+(theta_b-theta_c) for theta_a, theta_b, theta_c in zip(s['theta'][a], s['theta'][b],s['theta'][c]  )]
               elif evol_type=='ar':
                   s['phi'][d] = [phi_a + (phi_b - phi_c) for phi_a, phi_b, phi_c in zip(s['phi'][a], s['phi'][b], s['phi'][c])]
               elif evol_type=='r_var':
                   s['r_var'][d] = math.exp( math.log(s['r_var'][a]) + ( math.log(s['r_var'][b]) - math.log(s['r_var'][c])) )
               elif evol_type == 'q_var':
                   s['q_var'][d] = math.exp(math.log(s['q_var'][a]) + (math.log(s['q_var'][b]) - math.log(s['q_var'][c])))

        return s


    def _arma_matrix_update(s, k):
        n_states = s['n_states']
        n_obs = 1
        for ndx, (stl, ph, tht, r_var, q_var) in enumerate(
                zip(s['stale'], s['phi'], s['theta'], s['r_var'], s['q_var'])):
            p = len(ph)
            q = len(tht)
            if stl:
                A_ = np.zeros((n_states, n_states))
                A_[0, :p] = ph
                for j in range(n_states - 1):
                    A_[j + 1, j] = 1.
                s['A'][ndx, :, :] = A_
                H_ = np.ndarray((1,n_states))
                H_[0,:] = [1] + tht + [0 for _ in range(n_states-len(tht)-1)]
                s['H'][ndx, :, :] = H_
                Q_ = np.zeros((n_states, n_states))
                Q_[0, 0] = q_var
                s['Q'][ndx, :, :] = Q_
                R_ = np.ones((n_obs, n_obs))
                R_[0, 0] = r_var
                s['R'][ndx, :, :] = R_
                s['powers_of_A'][0, ndx, :, :] = A_
                for j in range(1, k):
                    s['powers_of_A'][j, ndx, :, :] = ddot(s['powers_of_A'][j - 1, ndx, :, :], A_)
        return s

    def _update(prior_mean, prior_covariance, observation_model, observation_noise, measurement, log_likelihood=False):

        # y - H * mp
        v = measurement - ddot(observation_model, prior_mean)

        # H * Pp * H.t + R
        S = ddot(observation_model, ddot_t_right(prior_covariance, observation_model)) + observation_noise
        invS = dinv(S)

        # Kalman gain: Pp * H.t * invS
        K = ddot(ddot_t_right(prior_covariance, observation_model), invS)

        # K * v + mp
        posterior_mean = ddot(K, v) + prior_mean

        # Pp - K * H * Pp
        posterior_covariance = prior_covariance - ddot(K, ddot(observation_model, prior_covariance))

        # inv-chi2 test var
        # outlier_test = np.sum(v * ddot(invS, v), axis=0)
        if log_likelihood:
            l = np.ravel(ddot(v.transpose((0,2,1)), ddot(invS, v)))
            l += np.log(np.linalg.det(S))
            l *= -0.5
            return posterior_mean, posterior_covariance, K, l
        else:
            return posterior_mean, posterior_covariance, K

        return posterior_mean, posterior_covariance