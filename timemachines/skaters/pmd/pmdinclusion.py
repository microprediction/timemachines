try:
    import pmdarima as pm
    using_pmd = True
except ImportError:
    using_pmd = False
    class Silly:
        nothing = True
    pm = Silly()


using_pmd = False   # Pending bug fix. See https://github.com/alkaline-ml/pmdarima/pull/455