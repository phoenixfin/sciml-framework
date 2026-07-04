# API reference, testing & docs tooling

## Generating the API reference

The codebase has full docstrings, so an API site can be generated from them with
[`pdoc`](https://pdoc.dev):

```bash
pip install -e ".[docs]"
pdoc sciml -o site/          # writes static HTML into site/
# or serve live:
pdoc sciml
```

> **Note on backends.** `pdoc` imports modules to introspect them, so generating
> docs for the TensorFlow engines (`deeponet`, `pinn`, `fno`, `neuralode`)
> requires TensorFlow installed (Python 3.10–3.12). The pure-numpy parts
> (`core`, `data`, `solvers`, `methods.sindy`, `methods.dmd`, all configs)
> document without a backend:
> ```bash
> pdoc sciml.core sciml.data sciml.solvers sciml.methods.sindy sciml.methods.dmd -o site/
> ```

## Docstring convention

All docstrings follow **NumPy style** (PEP 257 compliant): a one-line summary,
then `Parameters` / `Returns` / `Raises` sections that document **every**
parameter and the return value. Parameter types are written verbatim from the
function's type annotations (types live in the signature *and* the docstring so
the two stay in sync). Example:

```python
def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Relative L2 error ``||pred - ref|| / (||ref|| + eps)``.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    ref : np.ndarray
        Reference (ground-truth) values, same shape as ``pred``.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        The scalar relative L2 error.
    """
```

Config dataclass fields are documented with inline `#:` comments instead of an
`Attributes` block.

## Docstring coverage & completeness

Two complementary checks, both configured in `pyproject.toml` and run over
`src/`:

```bash
pip install -e ".[dev]"
interrogate src   # coverage: every public object HAS a docstring (fail-under = 100)
pydoclint src     # completeness: every parameter & return IS documented (0 violations)
```

- [`interrogate`](https://interrogate.readthedocs.io) (`[tool.interrogate]`)
  gates docstring **coverage** at 100% (trivial `__init__`/dunder/nested/private
  ignored).
- [`pydoclint`](https://jsh9.github.io/pydoclint/) (`[tool.pydoclint]`, NumPy
  style) checks docstring **completeness** — that documented parameters/returns
  match each signature. Zero violations across `src/`.

`pydocstyle` (`[tool.pydocstyle]`) can additionally spot-check PEP 257 style.

## Testing

```bash
pytest -q
```

- **Numpy tests always run** and are genuine end-to-end checks:
  - `test_examples_numpy.py` — SINDy *recovers* Lorenz, Van der Pol,
    FitzHugh–Nagumo coefficients from data; DMD recovers an oscillator frequency.
  - `test_solvers*.py` — conservation laws, positivity, a manufactured Darcy
    solution, KS boundedness.
  - `test_sindy.py`, `test_dmd.py`, `test_core.py`, `test_data.py`,
    `test_config.py` — unit behaviour.
- **TF tests skip cleanly** when TensorFlow is absent
  (`pytest.importorskip("tensorflow")` in `test_methods_tf.py`): DeepONet /
  FNO (1D+2D) / Neural ODE / grid-interp / SWE IC-shortcut shape & behaviour.

Because TensorFlow has no wheels for Python 3.13+/3.14, the TF engines are also
guarded by `python -m compileall src` (syntax check) in that environment.

## Continuous checks (suggested)

A minimal CI on a TF-free runner can enforce the always-available guarantees:

```bash
python -m compileall src        # syntax, incl. TF modules
pytest -q                       # numpy tests
interrogate src                 # 100% docstring coverage
```
