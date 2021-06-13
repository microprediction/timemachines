try:
    import divinity as dv
    using_divinity = True
except ImportError:
    class God:
        maybe = True
    dv = God()
    using_divinity = False