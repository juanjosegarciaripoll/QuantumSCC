# QuantumSCC:

## A Python Package for Automated Derivation and Diagonalization of Superconducting Circuit Hamiltonians

QuantumSCC is an open-source Python package designed to automate the derivation of
superconducting circuit Hamiltonians. The algorithm developed and implemented in the package is based on the 
approach proposed by Parra-Rodriguez and Egusquiza [1,2]. Furthermore, the program also incorporate advance tools 
for diagonalization of linear circuit Hamiltonians. Future work includes completing the diagonalization algorithm, making the package able to diagonalize nonlinear circuit Hamiltonians; introduce additional circuit components to the circuit configuration; and add the possible interaction between the circuit and an external electromagnetic field.

## Requirements

- Python >= 3.13
- Runtime dependencies (installed automatically): `numpy`, `scipy`, `sympy`

## Installation

QuantumSCC is a standard, [PEP 517](https://peps.python.org/pep-0517/)-compliant
Python package. It is not yet published on PyPI, so install it directly from the
repository.

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

```bash
# Install the latest version straight from GitHub
uv pip install "git+https://github.com/juanjosegarciaripoll/QuantumSCC.git"

# ...or, from a local clone
git clone https://github.com/juanjosegarciaripoll/QuantumSCC.git
cd QuantumSCC
uv pip install .
```

### Using a stock Python install (pip)

```bash
# From GitHub
python -m pip install "git+https://github.com/juanjosegarciaripoll/QuantumSCC.git"

# ...or, from a local clone
git clone https://github.com/juanjosegarciaripoll/QuantumSCC.git
cd QuantumSCC
python -m pip install .
```

Once installed, the package is importable as `QuantumSCC`:

```python
from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
```

## Development

The project uses a [`src/` layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/),
`pytest` for the test suite, and `ruff` for linting/formatting. Development
dependencies are declared in the `dev` dependency group of `pyproject.toml`.

### Using uv (recommended)

`uv` reads `pyproject.toml` and `uv.lock` to create a reproducible environment.

```bash
git clone https://github.com/juanjosegarciaripoll/QuantumSCC.git
cd QuantumSCC

# Create the virtual environment and install the package (editable) plus the
# dev dependency group, exactly as pinned in uv.lock
uv sync

# Run the test suite
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

### Using a stock Python install (pip + venv)

```bash
git clone https://github.com/juanjosegarciaripoll/QuantumSCC.git
cd QuantumSCC

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# Install the package in editable mode together with the dev tools
python -m pip install -e .
python -m pip install pytest ruff

# Run the test suite
pytest

# Lint and format
ruff check .
ruff format .
```

The test suite lives in the top-level `tests/` directory and is configured via
`[tool.pytest.ini_options]` in `pyproject.toml`.

## Examples
  - Linear circuit examples are available in: [examples_linear_circuits.ipynb](examples_linear_circuits.ipynb)
  - Nonlinear circuit examples are available in: [examples_nonlinear_circuits.ipynb](examples_nonlinear_circuits.ipynb)

## References
[1] A. Parra-Rodriguez and I. L. Egusquiza, Geometrical description and faddeev-jackiw quantization of electrical networks, Quantum 8, 1466 (2024).
[2] A. Parra-Rodriguez and I. L. Egusquiza, Exact quantization of nonreciprocal quasilumped electrical networks, Phys. Rev. X 15, 011072 (2025).

