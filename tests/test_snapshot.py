import shutil

import numpy as np

from snaptol.io import CACHE_KEY


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
    result.stdout.fnmatch_lines(["*Snapshot file not found*"])

    # Assert that the snapshot file is created.
    pytester.runpytest_subprocess("--snaptol-update").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

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
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()

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
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert not (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()

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
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    # Run the test only on test b.
    pytester.runpytest_subprocess(
        "-k", "test_b", "--snaptol-update-all"
    ).assert_outcomes(passed=1)

    # Check that test a snapshot was not deleted.
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()


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
    assert (pytester.path / "__snapshots__" / "test_abc.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.py__test_b.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.py__test_c.json").exists()

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
    assert (pytester.path / "__snapshots__" / "test_abc.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_abc.py__test_b.json").exists()
    assert not (pytester.path / "__snapshots__" / "test_abc.py__test_c.json").exists()


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
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

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
    assert not (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()


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
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

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
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()


def test_use_cache(pytester):
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([1, 2, 3], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Change the value.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([4, 5, 6], dtype=float))
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_a file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Allow the test to fail.
    pytester.runpytest_subprocess().assert_outcomes(failed=1)

    # Check that the cache was created.
    cache_dir = pytester.path / ".pytest_cache" / "v" / CACHE_KEY
    files = [p for p in cache_dir.glob("*") if p.is_file()]
    assert len(files) == 1
    cache_file = files[0]

    # Update the snapshots using the cache - we should therefore skip doing the test.
    pytester.runpytest_subprocess(
        "--snaptol-update", "--snaptol-use-cache"
    ).assert_outcomes(deselected=1)

    # Check the test now passes.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Check that the cache was deleted.
    assert not (cache_dir / cache_file).exists()


def test_delete_cache(pytester):
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([7, 8, 9], dtype=float))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Change the value.
    pytester.makepyfile(
        test_a="""
    import numpy as np
    def test_a(snaptolshot):
        snaptolshot.assert_allclose(np.array([10, 11, 12], dtype=float))
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_a file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Allow the test to fail.
    pytester.runpytest_subprocess().assert_outcomes(failed=1)

    # Check that the cache was created.
    cache_dir = pytester.path / ".pytest_cache" / "v" / CACHE_KEY
    files = [p for p in cache_dir.glob("*") if p.is_file()]
    assert len(files) == 1
    cache_file = files[0]

    # Do NOT update the snapshots using the cache.
    pytester.runpytest_subprocess("--snaptol-update").assert_outcomes(passed=1)

    # Check the test now passes.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Check that the cache was deleted despite not being used.
    assert not (cache_dir / cache_file).exists()


def test_show_diff(pytester):
    pytester.makepyfile(
        test_a="""
        def test_a(snaptolshot):
            assert snaptolshot == [1, 3]
        """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)

    # Add some values inbetween others.
    pytester.makepyfile(
        test_a="""
        def test_a(snaptolshot):
            assert snaptolshot == [1, 2, 3]
        """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_a file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Update the snapshot showing the difference and check it looks OK.
    result = pytester.runpytest_subprocess(
        "--snaptol-update-all", "--snaptol-show-diff"
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            " -+- snaptol diffs -+-",
            "--------------------------------------------------------------------------------",
            " Snapshot: *__snapshots__/test_a.py__test_a.json",
            "",
            "--- before",
            "+++ after",
            "@@ * @@",
            " [",
            "   1,",
            "+  2,",
            "   3",
        ],
        consecutive=True,
    )


def test_parameterise(pytester):
    # Create a test.
    pytester.makepyfile(
        test_a="""
    import pytest
    @pytest.mark.parametrize("parameter", [1, "a", True])
    def test_a(parameter, snaptolshot):
        result = parameter
        assert snaptolshot == result
    """
    )

    # Assert that the snapshot file is not found.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=3)

    # Assert that the snapshot files are created.
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a[1].json").exists()
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a[a].json").exists()
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a[True].json").exists()
