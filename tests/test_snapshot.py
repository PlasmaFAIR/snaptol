import numpy as np


def _get_dir_files(path):
    return [p.name for d in path.rglob("__snapshots__") for p in d.glob("*.json")]


def test_gaussian(snaptolshot):
    N = 100

    gaussian = np.exp(-(np.linspace(-5.0, 5.0, N) ** 2.0))

    # Normal tests.
    assert snaptolshot == gaussian

    # Don't do any more testing if we are updating the snapshot.
    if snaptolshot.snapshot_update:
        return

    # Normal tests continued.
    assert snaptolshot() == gaussian
    assert snaptolshot.match(gaussian)


def test_update_snapshot(pytester):
    # Create a test.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
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
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snaptolshot):
        assert snaptolshot == [1, 2, 3]
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
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
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
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snaptolshot):
        assert snaptolshot == [1, 2, 3]
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

    # Check that test a snapshot was not deleted.
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files


def test_remove_test_and_keyword(pytester):
    # Create 3 tests.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    def test_c(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=3)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files
    assert "test_ab.test_c.json" in files

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=3)

    # Remove test c.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """
    )

    # Run the test only on test b.
    pytester.runpytest_subprocess("-k", "test_b", "--snapshot-update").assert_outcomes(
        passed=1
    )

    # Check that test a was not deleted.
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" in files
    assert "test_ab.test_c.json" not in files


def test_remove_fixture(pytester):
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=1)
    files = _get_dir_files(pytester.path)
    assert "test_a.test_a.json" in files

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Keep the test but remove the fixture.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a():
        assert True
    """
    )

    # Update the snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=1)

    # Check that test a snapshot was deleted.
    files = _get_dir_files(pytester.path)
    assert "test_a.test_a.json" not in files


def test_skip(pytester):
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(passed=1)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Keep the test but skip it.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    import pytest
    @pytest.mark.skip
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Update the snapshots.
    pytester.runpytest_subprocess("--snapshot-update").assert_outcomes(skipped=1)

    # Check that test a snapshot was not deleted.
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
