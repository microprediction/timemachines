from timemachines.skaters.pmd.pmdinclusion import using_pmd
if using_pmd:
    from timemachines.skaters.pmd.pmdskaters import pmd_known, pmd_exogenous, pmd_univariate
    from timemachines.skaters.pmd.pmdcomposed import pmd_exogenous_hypocratic, pmd_univariate_hypocratic

    PMD_SKATERS = [pmd_univariate, pmd_exogenous_hypocratic]  # for now
else:
    PMD_SKATERS = []