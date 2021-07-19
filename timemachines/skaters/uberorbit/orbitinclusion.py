try:
    import orbit
    from orbit.models.lgt import LGTMAP, LGTAggregated, LGTFull
    using_orbit = True
except ImportError:
    LGTMAP = None
    LGTAggregated = None
    LGTFull = None
    using_orbit = False
