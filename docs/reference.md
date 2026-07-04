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

## Docstring coverage

Coverage is enforced with [`interrogate`](https://interrogate.readthedocs.io),
configured in `pyproject.toml` (`[tool.interrogate]`, `fail-under = 100`, with
trivial `__init__`/dunder/nested/private methods ignored):

```bash
pip install -e ".[dev]"
interrogate src            # -> RESULT: PASSED (100.0%)
interrogate -vv src        # per-object detail
```

Style (PEP 257) can be spot-checked with `pydocstyle` (config in
`[tool.pydocstyle]`):

```bash
pydocstyle src
```

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
