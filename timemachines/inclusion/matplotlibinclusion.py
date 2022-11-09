try:
    import matplotlib.pyplot as plt
    using_matplotlib = True
except ImportError:
    using_matplotlib = False
    plt = None