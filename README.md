# timemachines ⚠️ DEPRECATED

**`timemachines` is deprecated and has been replaced by [`skaters`](https://github.com/microprediction/skaters).**

`skaters` is a faster, lighter univariate forecasting package that runs in Pyodide. It completely supersedes `timemachines`.

![skating](https://i.imgur.com/elu5muO.png)

## Migrate

```bash
pip install skaters
```

```python
from skaters import laplace
```

## Compatibility shim

For now, `timemachines` is just a thin shim over `skaters`. It re-exports `laplace` so existing imports keep working, and emits a `DeprecationWarning`:

```python
from timemachines import laplace   # still works, but please switch to skaters
```

Everything else that used to live in `timemachines` (the `skaters`/`skatertools` subpackages, the per-package model adapters, evaluation tooling, etc.) has been removed. If you depend on any of it, pin an old release (`timemachines<0.21`) or, better, move to [`skaters`](https://github.com/microprediction/skaters).

## Links

- New package: https://github.com/microprediction/skaters
- Docs: https://skaters.microprediction.org/

MIT licensed.
