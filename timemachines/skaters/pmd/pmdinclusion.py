try:
    import pmdarima as pm
    using_pmd = True
except ImportError:
    using_pmd = False
    class Silly:
        nothing = True
    pm = Silly()
