from timemachines.skaters.uberorbit.orbitinclusion import using_orbit

if using_orbit:
    from timemachines.skaters.uberorbit.orbitlgtskaterfactory import orbit_lgt_12, orbit_lgt_24
    ORBIT_LGT_SKATERS = [orbit_lgt_12, orbit_lgt_24]
else:
    ORBIT_LGT_SKATERS = []


ORBIT_SKATERS = ORBIT_LGT_SKATERS