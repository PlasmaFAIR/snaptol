# snaptol

**Snapshot testing for numerical results with built in optional tolerance parameters.**

```snaptol``` is a pytest plugin for snapshot testing that comfortably handles tolerance with floating-point data.
It is designed for numerical, scientific, and simulation-based code, where results may often not be exactly deterministic and can depend on randomised initial conditions.

## Motivation

Consider an optimisation algorithm that may have a highly complex functional landscape or involve controlled randomness.
For example, Monte Carlo simulations, eigensolvers, or iterative solvers. A test may look like,
```python
import numpy as np

def test_something(snaptolshot):
    rng = np.random.default_rng(seed=123)
    random_initial_conditions = rng.normal(loc=0.0, scale=1e-6, size=100)
    result = very_complex_computation(random_initial_conditions)

    assert snaptolshot(rtol=1e-5, atol=1e-8) == result
```
```snaptol``` allows you to easily snapshot this result and provide tolerance parameters for future comparisons.
The plugin handles the storage and retrieval of data with JSON serialisation, and has support for NumPy arrays and its testing functions.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install snaptol.

```bash
pip install snaptol
```

## Usage

### Basic usage

In a test file, add the fixture, `snaptolshot`,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot == result
```
When `pytest` runs, it will compare the result to the snapshot stored in file.

### Using numerical tolerance
You can specify tolerances directly,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot(rtol=1e-05, atol=1e-08) == result
```

Or equivalently, use the `match` method if you prefer,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot.match(rtol=1e-05, atol=1e-08) == result
```

### Updating snapshots

#### Initial run (or intentional changes)

When creating snapshot tests for the first time, or when the expected result in a test has *intentionally* changed, run,
```
pytest --snaptol-update
```
This writes new snapshots instead of comparing values.

#### Important behaviour

The ```--snaptol-update``` flag will only update tests failed in the previous run. Use the ```--snaptol-update-all``` flag to enforce an update on all tests.

There should exist only one snapshot comparison per test. If the setup data is needed to be shared between different snapshot comparisons, then consider factoring it into a fixture or helper function.

### NumPy support

```snaptol``` is able to snapshot NumPy arrays and also integrates directly with the following testing utilities,

- assert_allclose
- assert_array_almost_equal_nulp
- assert_array_max_ulp
- assert_array_equal
- assert_equal
- assert_string_equal

For example,
```python
import numpy as np

def test_numpy_allclose(snaptolshot):
    result = np.arange(100, dtype=float)
    snaptolshot.assert_allclose(result)
```
Where any ```*args``` or ```**kwargs``` are passed directly to the corresponding NumPy function.

### Cache

```snaptol``` provides an optional caching feature that can be used during update runs.
This is particularly useful for testing suites that call expensive or time-consuming functions.

Results of tests that have failed on the previous run are stored in a cache directory on disk.
To enable caching, add the following option to an update run,
```
pytest --snaptol-update --snaptol-use-cache
```
When enabled, cached data is used in place of re-running the test.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)