from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, T_TYPE, E_TYPE
from typing import Any
from timemachines.skatertools.recommendations.suggestions import top_rated_models
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory
from timemachines.skatertools.utilities.internet import connected_to_internet


def elo_ensemble_factory(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None,r=None,
                         category='univariate',
                         max_seconds=1.0,
                         min_count=10,
                         require_passing=True,
                         n=5)->([float], Any, Any):
    """
        Ensemble of top performing skaters.

        r - determines variance weighting
        See top_rated_models for other argument meanings
    """
    assert r is not None, 'need the precision coef'
    if s.get('top_rated_skaters') is None:
        assert connected_to_internet(), 'cannot get best skaters without internet connection'
        top_rated_skaters = top_rated_models(k=k,n=n,category=category,max_seconds=max_seconds,min_count=min_count,
                                         require_passing=require_passing, ignore_elo=True)
        if len(top_rated_skaters)<=1:
            raise Exception('Could not find enough rated models')
        s['top_rated_skaters'] = top_rated_skaters


    return precision_weighted_ensemble_factory(fs=s['top_rated_skaters'], y=y, s=s, k=k, a=a, t=t, e=e, r=r)


