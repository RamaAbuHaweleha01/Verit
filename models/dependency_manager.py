#!/usr/bin/env python3
"""
Verit NIDS - Dependency Manager
------------------------------------------------------------------
Checks whether the packages required by the hybrid detection stack
(XGBoost + TensorFlow/Keras autoencoder) are present in the *current*
Python interpreter, and installs anything missing via `sys.executable
-m pip` -- so it always installs into the environment you actually
ran the script with (your venv), never a random system Python.

Call `ensure_dependencies()` once at the top of any script that needs
these models before importing xgboost/tensorflow.
"""

import importlib.util
import subprocess
import sys

# import name -> pip requirement spec
REQUIRED_PACKAGES = {
    "numpy": "numpy>=1.24.0",
    "pandas": "pandas>=2.0.0",
    "sklearn": "scikit-learn>=1.3.0",
    "joblib": "joblib>=1.3.0",
    "matplotlib": "matplotlib>=3.7.0",
    "xgboost": "xgboost>=2.0.0",
    "tensorflow": "tensorflow>=2.15.0",
    "flask": "flask>=3.0.0",
}


def _is_installed(import_name):
    return importlib.util.find_spec(import_name) is not None


def _pip_install(pip_spec):
    cmd = [sys.executable, "-m", "pip", "install", pip_spec]
    print(f"[deps] Installing {pip_spec} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and "externally-managed-environment" in (result.stderr or ""):
        # Not running inside a venv (PEP 668 system Python). Retry with the
        # override flag rather than failing outright -- this only matters
        # outside a venv; inside one (the normal case for this project),
        # the first attempt above already succeeds.
        print("[deps] System Python is externally managed; retrying with --break-system-packages ...")
        cmd = cmd + ["--break-system-packages"]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-3000:]
        raise RuntimeError(f"[deps] Failed to install {pip_spec}:\n{tail}")
    print(f"[deps] Installed {pip_spec}")


def ensure_dependencies(packages=None, quiet=False):
    """Check `packages` (default: REQUIRED_PACKAGES), install anything
    missing, then re-verify. Raises RuntimeError if a package still
    can't be imported after an install attempt."""
    packages = packages or REQUIRED_PACKAGES

    missing = [(imp, spec) for imp, spec in packages.items() if not _is_installed(imp)]
    if not missing:
        if not quiet:
            print("[deps] All required packages already installed.")
        return

    print(f"[deps] Missing packages detected: {[imp for imp, _ in missing]}")
    print(f"[deps] Installing into: {sys.executable}")
    for imp, spec in missing:
        _pip_install(spec)

    still_missing = [imp for imp, _ in missing if not _is_installed(imp)]
    if still_missing:
        raise RuntimeError(
            f"[deps] These packages are still missing after installation attempts: {still_missing}. "
            f"Install them manually with: {sys.executable} -m pip install <package>"
        )

    if not quiet:
        print("[deps] All required packages are now available.")


if __name__ == "__main__":
    ensure_dependencies()
