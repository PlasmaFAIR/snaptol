import numpy as np


def _get_dir_files(path):
    return [p.name for d in path.rglob("__snapshots__") for p in d.glob("*.json")]


def test_gaussian(snaptolshot):
    N = 100

    gaussian = np.exp(-(np.linspace(-5.0, 5.0, N) ** 2.0))

    # Normal tests.
    assert snaptolshot == gaussian

    # Don't do any more testing if we are updating the snapshot.
    if snaptolshot.snaptol_update:
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
    pytester.runpytest_subprocess("--snaptol-update").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.test_a.json").exists()

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
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert (pytester.path / "__snapshots__" / "test_ab.test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.test_b.json").exists()

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

    # Remove the cache to absolutely ensure Python runs on the overwritten test_ab file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Update the snapshots - should delete snapshot file b.
    pytester.runpytest_subprocess("--snaptol-update").assert_outcomes(passed=1)
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert "test_ab.test_b.json" not in files
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_ab.test_a.json").exists()
    assert not (pytester.path / "__snapshots__" / "test_ab.test_b.json").exists()

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
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert (pytester.path / "__snapshots__" / "test_ab.test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.test_b.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    # Run the test only on test b.
    pytester.runpytest_subprocess(
        "-k", "test_b", "--snaptol-update-all"
    ).assert_outcomes(passed=1)

    # Check that test a snapshot was not deleted.
    assert (pytester.path / "__snapshots__" / "test_ab.test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.test_b.json").exists()


def test_remove_test_and_keyword(pytester):
    # Create 3 tests.
    pytester.makepyfile(
        test_abc="""
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
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=3)
    assert (pytester.path / "__snapshots__" / "test_abc.test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.test_b.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.test_c.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=3)

    # Remove test c.
    pytester.makepyfile(
        test_abc="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    def test_b(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_ab file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Run the test only on test b.
    pytester.runpytest_subprocess(
        "-k", "test_b", "--snaptol-update-all"
    ).assert_outcomes(passed=1)

    # Check that test a was not deleted.
    assert (pytester.path / "__snapshots__" / "test_abc.test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.test_b.json").exists()
    assert not (pytester.path / "__snapshots__" / "test_abc.test_c.json").exists()


def test_remove_fixture(pytester):
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.test_a.json").exists()

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

    # Remove the cache to absolutely ensure Python runs on the overwritten test_a file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Update the snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)

    # Check that test a snapshot was deleted.
    assert not (pytester.path / "__snapshots__" / "test_a.test_a.json").exists()


def test_skip(pytester):
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.test_a.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Keep the test but skip it.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    import pytest
    @pytest.mark.skip
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_ab file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Update the snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(skipped=1)

    # Check that test a snapshot was not deleted.
    files = _get_dir_files(pytester.path)
    assert "test_ab.test_a.json" in files
    assert (pytester.path / "__snapshots__" / "test_a.test_a.json").exists()

