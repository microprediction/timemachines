from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
from typing import Any, List, Union
from timemachines.skatertools.ensembling.ensemblefactory import ensemble_factory
from timemachines.skatertools.crowd.lotteryconventions import E_SUGGESTION, E_REWARD_STR, EXTENDED_E_TYPE
from timemachines.skatertools.recommendations.hardwiredsuggestions import U_1_5
from timemachines.inclusion.threezainclusion import using_threeza
import json

if using_threeza:
    from threeza.crowd.ongoingcategoricallottery import OngoingCategoricalLottery
    from threeza.conventions import horizon_str_to_k_and_tau
    from timemachines.skaters.localskaters import local_skater_from_name, LOCAL_SKATERS
    from momentum import var_init, var_update
    import numpy as np

# --------------------------------------------------------------------------------------------
# A stacking of local skaters where exogenous suggestions can be rewarded.
#
# The difference between this an other autonomous ensembles is that the skater has a mechanism
# for receiving recommendations of skaters it might try out on the next epoch.
# -------------------------------------------------------------------------------------------




# Default list of pretty fast skaters
DEFAULT_LOTTERY_FS = U_1_5[:3]
DEFAULT_LOTTERY_FA = U_1_5


def lottery_ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: EXTENDED_E_TYPE = None,
                     g=None, r=None, include_std=True, fa=None) -> ([float], Any, Any):
    """ Crowd-informed stacking of skaters

              fa  - larger list of skaters that we are open to using
              g   - exogenous skater
              r   - hyper-param for g, if any
              include_std - bool. If True, will add x_std to the exogenous variables sent to g

        :returns
                reward dict if e is str
                s      if e is dict

        This skater starts by building a stacked model using DEFAULT_LOTTERY_FS but it can also receive
        'suggestions' for skaters, which are really predictions of which skater will be the winner at the
        time when a reward is announced.

        This uses a dict-like state object

    """
    if not s.get('crowd'):
        s = {'s_crowd':OngoingCategoricalLottery(allowed_values=fa, allowed_horizons=['k=1&tau=-3']),
             's_ensemble':{}
             }

    if isinstance(e,dict) and 'owner' in e:
        # Expects a suggestion
        #  owner:   Identifier for person making the suggestion of which skaters will perform best
        #  values:  A list of skater names
        #  weights: A list of winning probabilities for the corresponding skaters (optional)
        #  amount:  A measure of confidence in the prediction (bet amount)
        assert t is not None, 'Need t when crowd prediction is received '
        values = e['values']
        owner = e['owner']
        weights = e.get('weights')
        amount = e.get('amount')
        if amount is None:
            amount = 1.0
        s['s_crowd'].add(t=t, owner=owner, values=values, weights=weights, amount=amount)
        return s
    elif isinstance(e,str):
        # Reward signal triggers a determination of the most accurate skater
        assert e==E_REWARD_STR, 'Expecting to see ' + E_REWARD_STR
        s['reward'] = s['s_crowd'].payout(t=t, value=s['s_ensemble']['s_w'], consolidate=True )
    if y is not None:
        if s.get('fs') is None:
            s['fs'] = [nm for nm in DEFAULT_LOTTERY_FS]
        x, x_std, s['s_ensemble'] = ensemble_factory(y=y, s=s.get('s_ensemble'), k=k, a=a, t=t, e=e, fs=s['fs'], rs=None, g=g, r=r, include_std=include_std)

        return x, x_std, s


def lottery_ensemble_state_from_json(s:str):
    """
        Helper to inflate the lottery ensemble state
    """
    # (for the other direction just use regular json.dumps() to serialize)
    s1 = json.loads(s)
    s1['s_crowd'] = OngoingCategoricalLottery(**json.loads(s1['s_crowd']))
    return s1

