# Release checks

As checked on 2026-10-30, PyPI 2.1.0 (2026-07-08) still contains the old
`min(p_m, 2*p_n)` combination. Repository source corrects it to
`min(1, 2*min(p_m, p_n))` once the second channel is armed. See #138 and #167.
The version in `pyproject.toml` is still 2.1.0; building a wheel does not make
that wheel the existing PyPI artifact or authorize publishing it.

Before an explicitly approved release:

1. Review open release blockers, including browser parity (#143) and calibration
   provenance (#163). Record the exact source commit and dependency versions.
2. Select a new unused version and build the sdist and wheel from the reviewed
   commit. Do not reuse the published 2.1.0 version.
3. Install the built wheel into a fresh environment and execute the regression
   **outside pytest**, whose conftest adds the checkout's `src/` to the import path:

   ```sh
   python -m build
   python -m venv /tmp/timemachines-wheel
   /tmp/timemachines-wheel/bin/python -m pip install dist/*.whl
   /tmp/timemachines-wheel/bin/python -I scripts/smoke_installed_wheel.py
   ```

   Use a new empty build directory/environment for each candidate. CI and the
   publish workflow run this gate before upload. It checks the scripted-state
   alarm decision from #167 and the unarmed/armed combination contract, not
   statistical calibration or optional detector integrations.
4. Run the complete source tests and inspect all CI results. Calibrate and record
   future tables against a named version, source commit, dependency versions and
   evaluation protocol; do not carry old tables forward as new validation.
5. Only after release approval, publish through the existing release process.
   Download the newly published wheel into another clean environment and rerun
   the same smoke check before removing the README/homepage release caveats.

This checklist and the smoke gate do not publish anything. A correction to
combination arithmetic does not establish calibrated marginal p-values or
safe production deployment. Keep the released and unreleased behavior explicit.
