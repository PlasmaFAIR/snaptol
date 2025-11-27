# snaptol

A Python tool for snapshot testing with numerical tolerance on floating point numbers.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install snaptol.

```bash
pip install snaptol
```

## Usage

### Normal usage

In a test file, add the snapshot fixture, `snaptolshot`,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot == result
```
When `pytest` runs, it will compare the result to the snapshot stored in file.

To provide a tolerance, pass it as an argument to the fixture,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot(rtol=1e-05, atol=1e-08) == result
```

Alternatively, use the `match` method,
```python
def test_something(snaptolshot):
    result = compute_something()
    assert snaptolshot.match(rtol=1e-05, atol=1e-08) == result
```

### Updating snapshots

On initial pass and subsequent changes to the test, run `pytest` with the `--snapshot-update` flag,
```python
pytest --snapshot-update
```
This will not perform the comparison between snapshot and result, but rather update the snapshot file with the current result.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)