import numpy as np
import numpy.random as npr
import pytest


def test_numpy_allclose(snaptolshot):
    # Example test case of a damped oscillator.
    N = 100
    t = np.linspace(0, 10, N, dtype=float)
    y = np.exp(-0.1 * t) * np.sin(2.0 * np.pi * t)

    generator = npr.default_rng()

    def noise(size):
        return size * (2.0 * generator.random(N) - 1.0)

    snaptolshot.assert_allclose(y)

    if snaptolshot.snapshot_update:
        return

    # Add some SMALL random noise.
    y_low_noise = y + noise(1e-8)

    # Should all still be OK.
    snaptolshot.assert_allclose(y_low_noise, rtol=1e-8, atol=1e-8)

    # Add some BIG random noise.
    y_high_noise = y + noise(1e-7)

    with pytest.raises(AssertionError):
        snaptolshot.assert_allclose(y_high_noise, rtol=1e-8, atol=1e-8)


def test_numpy_array_almost_equal_nulp(snaptolshot):
    x = np.array([1, 1e-10, 1e-20], dtype=float)
    y = np.nextafter(x, np.inf)  # Only 1 unit in last place (ULP) after x - OK.
    z = np.nextafter(y, np.inf)  # Now 2 ULPs after x - NOT OK.

    nulp = 1

    snaptolshot.assert_array_almost_equal_nulp(x, nulp=nulp)

    if snaptolshot.snapshot_update:
        return

    snaptolshot.assert_array_almost_equal_nulp(y, nulp=nulp)

    with pytest.raises(AssertionError):
        snaptolshot.assert_array_almost_equal_nulp(z, nulp=nulp)


def test_numpy_array_max_ulp(snaptolshot):
    x = np.array([2, 2e-10, 2e-20], dtype=float)
    y = x
    maxulp = 5

    snaptolshot.assert_array_max_ulp(x, maxulp=maxulp)

    if snaptolshot.snapshot_update:
        return

    for nulp in range(1, 10):
        y = np.nextafter(y, np.inf)

        if nulp <= maxulp:
            snaptolshot.assert_array_max_ulp(y, maxulp=maxulp)
        else:
            with pytest.raises(AssertionError):
                snaptolshot.assert_array_max_ulp(y, maxulp=maxulp)


def test_numpy_array_equal(snaptolshot):
    x = np.array([1.0, 10.0, 100.0], dtype=float)

    snaptolshot.assert_array_equal(x)

    if snaptolshot.snapshot_update:
        return

    y = np.array([2.0, 20.0, 200.0], dtype=float)

    with pytest.raises(AssertionError):
        snaptolshot.assert_array_equal(y)


def test_numpy_equal(snaptolshot):
    x = np.array([3, 3, 3])

    snaptolshot.assert_equal(x)


def test_numpy_string_equal(snaptolshot):
    x = "abc"

    snaptolshot.assert_string_equal(x)

    if snaptolshot.snapshot_update:
        return

    with pytest.raises(AssertionError):
        snaptolshot.assert_string_equal("def")
