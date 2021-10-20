
try:
    import merlion
    using_merlion = True
except ImportError:
    using_merlion = False

if __name__=='__main__':
    print(using_merlion)