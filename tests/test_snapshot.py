import shutil
from textwrap import dedent

import numpy as np
import pytest

from snaptol.io import CACHE_KEY


@pytest.fixture
def gaussian():
    N = 100
    return np.exp(-(np.linspace(-5.0, 5.0, N) ** 2.0))


def test_assert(snaptolshot, gaussian):
    assert snaptolshot == gaussian


def test_call_assert(snaptolshot, gaussian):
    assert snaptolshot() == gaussian


def test_assert_match(snaptolshot, gaussian):
    assert snaptolshot.match(gaussian)


def test_multiple_asserts(snaptolshot):
    assert snaptolshot == 1.1
    assert snaptolshot == 2.2
    assert snaptolshot == 3.3


def test_multiple_named_asserts(snaptolshot):
    assert snaptolshot["a"] == 1.1
    assert snaptolshot["b"] == 2.2
    assert snaptolshot["c"] == 3.3


def test_multiple_named_numpy_asserts(snaptolshot):
    assert snaptolshot["one"].assert_allclose([1.1, 1.2, 1.3])
    assert snaptolshot["two"].assert_allclose([2.2, 2.2, 2.3])


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
    result.stdout.fnmatch_lines(["*Snapshot file '*' not found*"])

    # Assert that the snapshot file is created.
    pytester.runpytest_subprocess("--snaptol-update").assert_outcomes(passed=1)
    assert (pytester.path / "__snapshots__" / "test_a.py__test_a.json").exists()

    # Assert that the snapshot check passes.
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_remove_test(pytester):
    """Check that removing a test removes its snapshots, including from multiple asserts"""

    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]

    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
        assert snaptolshot == [7, 8, 9]
    """
    )

    # Create snapshots.
    base_test_path = pytester.path / "__snapshots__" / "test_ab"
    base_test_path.mkdir(parents=True, exist_ok=True)
    (base_test_path / "test_a.json").write_text("[1, 2, 3]")
    (base_test_path / "test_b.json").write_text("[4, 5, 6]")
    (base_test_path / "test_b[1].json").write_text("[7, 8, 9]")

    # Rewrite the file to delete test b.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_ab file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Update the snapshots - should delete snapshot file b.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    base_test_path = pytester.path / "__snapshots__"
    assert (base_test_path / "test_ab.py__test_a.json").exists(), (
        "Snapshot for test a doesn't exist"
    )
    assert not (base_test_path / "test_ab.py__test_b.json").exists(), (
        "Snapshot for test b first assert not removed"
    )
    assert not (base_test_path / "test_ab.py__test_b[1].json").exists(), (
        "Snapshot for test b second assert not removed"
    )

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


@pytest.mark.parametrize(
    "skip_marker",
    ["@pytest.mark.skip", "@pytest.mark.skipif(True, reason='skipped')"],
)
@pytest.mark.parametrize("snapshot_dir", ["__snapshots__", "__custom__"])
def test_skip_alongside_other_test(pytester, skip_marker, snapshot_dir):
    # When another test runs in the same directory, the snapshot files in that
    # directory get checked, so the skipped test's files must be recognised
    # Also checks the marker applies at module level
    dir_marker = (
        ""
        if snapshot_dir == "__snapshots__"
        else f"pytest.mark.snaptol_dir('{snapshot_dir}')"
    )
    test_ab = """
    import pytest
    pytestmark = [{dir_marker}]
    {skip_marker}
    def test_a(snaptolshot):
        assert snaptolshot[0] == [1, 2, 3]
        assert snaptolshot[1] == [4, 5, 6]
    def test_b(snaptolshot):
        assert snaptolshot == [7, 8, 9]
    """
    pytester.makepyfile(test_ab=test_ab.format(dir_marker=dir_marker, skip_marker=""))
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert (pytester.path / snapshot_dir / "test_ab.py__test_a.json").exists()

    pytester.makepyfile(
        test_ab=test_ab.format(dir_marker=dir_marker, skip_marker=skip_marker)
    )
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.no_fnmatch_line("*could be deleted*")

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(
        passed=1, skipped=1
    )
    assert (pytester.path / snapshot_dir / "test_ab.py__test_a.json").exists()
    assert (pytester.path / snapshot_dir / "test_ab.py__test_a[1].json").exists()
    assert (pytester.path / snapshot_dir / "test_ab.py__test_b.json").exists()


def test_skip_in_test_body(pytester):
    pytester.makepyfile(
        test_ab="""
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)

    pytester.makepyfile(
        test_ab="""
    import pytest
    def test_a(snaptolshot):
        pytest.skip()
        assert snaptolshot == [1, 2, 3]
    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(
        passed=1, skipped=1
    )
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()


def test_deselected_custom_snapshot_dir(pytester):
    custom_dir = "__custom_snapshot_dir__"
    pytester.makepyfile(
        test_ab=f"""
    import pytest
    @pytest.mark.snaptol_dir("{custom_dir}")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    @pytest.mark.snaptol_dir("{custom_dir}")
    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)

    pytester.runpytest_subprocess(
        "--snaptol-update-all", "-k", "test_b"
    ).assert_outcomes(passed=1, deselected=1)
    assert (pytester.path / custom_dir / "test_ab.py__test_a.json").exists()
    assert (pytester.path / custom_dir / "test_ab.py__test_b.json").exists()


def test_remove_fixture_custom_snapshot_dir(pytester):
    # Each test has its own custom directory, so test_a's directory is only
    # known about via its marker
    custom_dir_a = pytester.path / "__custom_a__"
    custom_dir_b = pytester.path / "__custom_b__"
    pytester.makepyfile(
        test_ab="""
    import pytest
    @pytest.mark.snaptol_dir("__custom_a__")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    @pytest.mark.snaptol_dir("__custom_b__")
    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert (custom_dir_a / "test_ab.py__test_a.json").exists()
    assert (custom_dir_b / "test_ab.py__test_b.json").exists()

    # Check the snapshots pass.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    # Remove the fixture from test_a, but keep its marker.
    pytester.makepyfile(
        test_ab="""
    import pytest
    @pytest.mark.snaptol_dir("__custom_a__")
    def test_a():
        assert True
    @pytest.mark.snaptol_dir("__custom_b__")
    def test_b(snaptolshot):
        assert snaptolshot == [4, 5, 6]
    """
    )
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Normal mode -> only suggest deleting test_a's snapshot.
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*could be deleted*", "*test_ab.py__test_a.json"])
    assert (custom_dir_a / "test_ab.py__test_a.json").exists()

    # Update mode -> test_a's snapshot (and now empty directory) is removed.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert not (custom_dir_a / "test_ab.py__test_a.json").exists()
    assert not custom_dir_a.exists()
    assert (custom_dir_b / "test_ab.py__test_b.json").exists()


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
            "-[1, 3]",
            "+[1, 2, 3]",
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


def test_parameterise_multiple_asserts(pytester):
    # Create a test.
    pytester.makepyfile(
        test_a="""
    import pytest
    @pytest.mark.parametrize("parameter1, parameter2", [(1, 2), ("a", "b"), (True, False)])
    def test_a(parameter1, parameter2, snaptolshot):
        assert snaptolshot == parameter1
        assert snaptolshot == parameter2
    """
    )

    # Assert that the snapshot file is not found.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=3)

    # Assert that the snapshot files are created.
    base_test_path = pytester.path / "__snapshots__"
    assert (base_test_path / "test_a.py__test_a[1-2].json").exists()
    assert (base_test_path / "test_a.py__test_a[a-b].json").exists()
    assert (base_test_path / "test_a.py__test_a[True-False].json").exists()
    assert (base_test_path / "test_a.py__test_a[1-2][1].json").exists()
    assert (base_test_path / "test_a.py__test_a[a-b][1].json").exists()
    assert (base_test_path / "test_a.py__test_a[True-False][1].json").exists()


def test_compare_different_types(pytester):
    # Create a test for strings.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        assert snaptolshot == np.array(['h', 'i'], dtype='<U1')
    def test_b(snaptolshot):
        assert snaptolshot.assert_allclose(np.array(['h', 'i'], dtype='<U1'))
    """
    )

    # Create snapshots.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_a.json").exists()
    assert (pytester.path / "__snapshots__" / "test_ab.py__test_b.json").exists()

    # Rewrite the test for floats - the comparison will fail.
    pytester.makepyfile(
        test_ab="""
    import numpy as np
    def test_a(snaptolshot):
        assert snaptolshot == np.array([0.0, 1.0], dtype='float64')
    def test_b(snaptolshot):
        assert snaptolshot.assert_allclose(np.array([0.0, 1.0], dtype='float64'))
    """
    )

    # Remove the cache to absolutely ensure Python runs on the overwritten test_a file and not the original.
    shutil.rmtree(pytester.path / "__pycache__", ignore_errors=True)

    # Normal mode -> should fail as we have a type incompatibility.
    pytester.runpytest_subprocess().assert_outcomes(failed=2)

    # Update mode -> should pass despite the type incompatibility.
    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=2)

    # Normal mode -> should now pass as there should no longer be a type incompatibility.
    pytester.runpytest_subprocess().assert_outcomes(passed=2)


def test_custom_snapshot_dir(pytester):
    custom_dir = "__custom_snapshot_dir__"
    pytester.makepyfile(
        test_a=f"""
    import pytest
    @pytest.mark.snaptol_dir("{custom_dir}")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
        assert snaptolshot == [4, 5, 6]
    """
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    # Check we've created snapshots in our custom location
    assert (pytester.path / custom_dir).exists()

    # Remove an assert and check we remove the associated file
    pytester.makepyfile(
        test_a=f"""
    import pytest
    @pytest.mark.snaptol_dir("{custom_dir}")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)

    assert (pytester.path / custom_dir / "test_a.py__test_a.json").exists()
    assert not (pytester.path / custom_dir / "test_a.py__test_a[1].json").exists()


def test_custom_snapshot_dir_absolute_path(pytester):
    test_dir = pytester.path / "tests_dir"
    test_dir.mkdir()

    custom_dir = pytester.path / "elsewhere" / "__custom_snapshot_dir__"
    (test_dir / "test_a.py").write_text(
        dedent(f"""
    import pytest
    @pytest.mark.snaptol_dir(r"{custom_dir}")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
        assert snaptolshot == [4, 5, 6]
    """)
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    # Check we've created snapshots in our custom location
    assert custom_dir.exists()

    # Remove an assert and check we remove the associated file
    (test_dir / "test_a.py").write_text(
        dedent(f"""
    import pytest
    @pytest.mark.snaptol_dir(r"{custom_dir}")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """)
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)

    assert (custom_dir / "test_a.py__test_a.json").exists()
    assert not (custom_dir / "test_a.py__test_a[1].json").exists()


def test_custom_snapshot_dir_relative_to_test_file(pytester):
    test_dir = pytester.path / "tests_dir"
    test_dir.mkdir()
    (test_dir / "test_a.py").write_text(
        dedent("""
    import pytest
    @pytest.mark.snaptol_dir("__custom_snapshot_dir__")
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """)
    )

    pytester.runpytest_subprocess("--snaptol-update-all").assert_outcomes(passed=1)
    assert (test_dir / "__custom_snapshot_dir__" / "test_a.py__test_a.json").exists()
    assert not (pytester.path / "__custom_snapshot_dir__").exists()


def test_custom_default_snapshot_dir(pytester):
    custom_dir = "__custom_snapshot_dir__"
    pytester.makepyfile(
        test_a="""
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """
    )

    pytester.runpytest_subprocess(
        "--snaptol-update-all", f"--snaptol-dirname={custom_dir}"
    ).assert_outcomes(passed=1)
    # Check we've created snapshots in our custom location
    assert (pytester.path / custom_dir).exists()

    # Remove an assert an check we remove the associated file
    pytester.makepyfile(
        test_a="""
    def test_a():
        assert True
    """
    )

    pytester.runpytest_subprocess(
        "--snaptol-update-all", f"--snaptol-dirname={custom_dir}"
    ).assert_outcomes(passed=1)

    assert not (pytester.path / custom_dir).exists()


def test_custom_default_snapshot_dir_absolute_path(pytester):
    test_dir = pytester.path / "tests_dir"
    test_dir.mkdir()

    custom_dir = pytester.path / "elsewhere" / "__custom_snapshot_dir__"
    (test_dir / "test_a.py").write_text(
        dedent("""
    def test_a(snaptolshot):
        assert snaptolshot == [1, 2, 3]
    """)
    )

    pytester.runpytest_subprocess(
        "--snaptol-update-all", f"--snaptol-dirname={custom_dir}"
    ).assert_outcomes(passed=1)
    # Check we've created snapshots in our custom location
    assert custom_dir.exists()

    # Remove an assert and check we remove the associated file
    (test_dir / "test_a.py").write_text(
        dedent("""
    def test_a():
        assert True
    """)
    )

    pytester.runpytest_subprocess(
        "--snaptol-update-all", f"--snaptol-dirname={custom_dir}"
    ).assert_outcomes(passed=1)

    assert not custom_dir.exists()
