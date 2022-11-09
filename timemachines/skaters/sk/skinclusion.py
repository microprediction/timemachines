
try:
    import sktime
    using_sktime = True
except ImportError:
    using_sktime = False

if __name__=='__main__':
    print({'using_sktime':using_sktime})