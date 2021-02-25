from timemachines.skatertools.utilities.conventions import wrap, split_exogenous


# Let's get on thing straight. An observance is *not* a collection of hermits. Not here anyway.
# Rather, it is a simple cache of
#    - The most recent exogenous variables y[1:]
#    - A queue of k-step ahead "a" variables that are waiting to slide back to the present
# Recall that in the skater convention, "a" variables are supplied k-steps in advance so that
# k-step ahead prediction is possible. But this means we have to cache them until they become
# contemporaneous with y[1:]. Only then can we create an aligned time series of regressors and
# regressand.


def observance(y:[float], o:dict, k:int, a:[float]=None):
    """
    This marshals the k-step ahead vector a and the contemporaneous y[1:] and
    returns a combined vector of all exogenous variables.

    It tracks a list of x and corresponding y, by putting a's in a FIFO queue and
    by caching the previous value of y[1:]

    :param o:  state
    :param k:  Number of steps ahead that a is provided
    :param y:
    :param a:
    :returns:  x_t:[float] vector combining y[1:] with previously supplied a's
    """
    yw = wrap(y)
    aw = wrap(a)

    if not o:
        o = {'a':[ None for _ in range(k) ],
             'z':None,  # Stores the previous value of y[1:]
             'x':list(),
             'y':list()
             }

    y_t, z = split_exogenous(yw)

    # Get the contemporaneous variables from last observation
    if z:
        z_t = o.get('z')  # The previously revealed exogenous variables
        o['z'] = z        # Store for next time
    else:
        z = None
        z_t = None

    # Determine the known in advance variable pertaining to the present
    if aw:
        a_t = o['a'].pop(0)  # The known in advance variable pertaining to this time step
        o['a'].append(aw)  # Put the k-ahead received a value(s) on the queue
    else:
        a = None
        a_t = None

    # Combine into exogenous variables ... but only if both arrived
    if aw and z:
        x_t = z_t + a_t if (z_t and a_t) else None
    elif aw and not z:
        x_t = a_t if a_t else None
    elif (not aw) and z:
        x_t = z_t if z_t else None
    elif (not aw) and not z:
        x_t = None

    if (not z) and (not aw):
        o['y'].append([y_t])  # Special case, no need to wait
    else:
        if x_t:
            o['x'].append(x_t)
            o['y'].append([y_t])
        assert len(o['x']) == len(o['y']), "post-condition"
    return x_t, o

