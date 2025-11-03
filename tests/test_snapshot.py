import numpy as np
import numpy.random as npr
import pytest


def _get_dir_files(path):
    return [p.name for d in path.rglob("__snapshots__") for p in d.glob("*.json")]


def test_gaussian(snapshot):
    N = 100
    generator = npr.default_rng()

    def noise(size):
        return size * (2.0 * generator.random(N) - 1.0)

    gaussian = np.exp(-(np.linspace(-5.0, 5.0, N) ** 2.0))

    # Normal tests.
    assert snapshot == gaussian

    # Don't do any more testing if we are updating the snapshot.
    if snapshot.snapshot_update:  # TODO: is this logic OK inside of a test?
        return

    # Normal tests continued.
    assert snapshot() == gaussian
    assert snapshot.match(gaussian)
    snapshot.assert_allclose(gaussian)

    # Add some SMALL random noise.
    gaussian_low_noise = gaussian + noise(1e-8)

    # Should all still be OK.
    assert snapshot == gaussian_low_noise
    assert snapshot() == gaussian_low_noise
    assert snapshot.match(gaussian_low_noise)
    snapshot.assert_allclose(
        gaussian_low_noise, atol=1e-8
    )  # atol defined here as NumPy has different default tolerances.

    # Add some BIG random noise.
    gaussian_high_noise = gaussian + noise(1e-7)

    # Should now fail as noise is too large.
    with pytest.raises(AssertionError):
        assert snapshot == gaussian_high_noise
    with pytest.raises(AssertionError):
        assert snapshot() == gaussian_high_noise
    with pytest.raises(AssertionError):
        assert snapshot.match(gaussian_high_noise)
    with pytest.raises(AssertionError):
        snapshot.assert_allclose(
            gaussian_high_noise, atol=1e-8
        )  # atol defined here as NumPy has different default tolerances.


def test_update_snapshot(pytester):
    # Create a test.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snapshot):
        snapshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Assert that the snapshot file is not found.
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Snapshot not found*"])

    # Assert that the snapshot file is created.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=1)
    assert "test_a.test_a.json" in _get_dir_files(pytester.path)

    # Assert that the snapshot check passes.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_remove_test(pytester):
    # Create 2 tests.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snapshot):
        snapshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snapshot):
        snapshot == [1, 2, 3]
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=2)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    # Rewrite the file to delete test b.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snapshot):
        snapshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Update the snapshots - should delete snapshot file b.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=1)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" not in files

    # Check snapshot a still passes.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_keyword(pytester):
    # Create 2 tests.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snapshot):
        snapshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snapshot):
        snapshot == [1, 2, 3]
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=2)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    # Run the test only on test b.
    pytester.runpytest_subprocess("-k", "test_b", "--snapshot-update").assert_outcomes(
        passed=1
    )

    # Check that test b was not deleted.
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files
