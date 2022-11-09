try:
    import matplotlib.pyplot as plt
    using_matplotlib = True
except Exception as e: # Could be different kinds of error
    using_matplotlib = False
    plt = None