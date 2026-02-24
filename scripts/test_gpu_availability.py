#!/usr/bin/env python3
"""Tests for _check_plain_host_availability GPU memory parsing logic."""

from unittest.mock import MagicMock, patch


def make_run_result(stdout: str, ok: bool = True) -> MagicMock:
    r = MagicMock()
    r.ok = ok
    r.stdout = stdout
    r.stderr = ""
    return r


def check(nvidia_smi_output: str, configured: int = 4) -> int:
    from theseus.dispatch.solve import _check_plain_host_availability

    with patch(
        "theseus.dispatch.ssh.run", return_value=make_run_result(nvidia_smi_output)
    ):
        return _check_plain_host_availability("fake-host", configured, timeout=5.0)


def test_no_processes():
    assert check("") == 4, "empty output → all chips free"


def test_xorg_only():
    # wroclaw-style: Xorg at 4 MiB on each GPU
    output = (
        "GPU-abc, 3374, /usr/lib/xorg/Xorg, 4\nGPU-def, 3374, /usr/lib/xorg/Xorg, 4\n"
    )
    assert check(output) == 4, "Xorg at 4 MiB → noise, chips free"


def test_zero_memory():
    output = "GPU-abc, 1234, some-daemon, 0\n"
    assert check(output) == 4, "zero memory → noise, chips free"


def test_na_memory():
    # fused arch (GB10 etc.) reports N/A
    output = "GPU-abc, 1234, some-process, N/A\n"
    assert check(output) == 4, "N/A memory → noise, chips free"


def test_missing_memory_field():
    # only 3 fields — memory column absent entirely
    output = "GPU-abc, 1234, some-process\n"
    assert check(output) == 4, "missing field → noise, chips free"


def test_real_training_job():
    output = "GPU-abc, 9999, python, 38000\n"
    assert check(output) == 0, "38 GiB usage → busy"


def test_mixed_noise_and_real():
    output = "GPU-abc, 3374, /usr/lib/xorg/Xorg, 4\nGPU-def, 9999, python, 38000\n"
    assert check(output) == 0, "one real job → busy despite noise"


def test_exactly_at_threshold():
    output = "GPU-abc, 9999, python, 100\n"
    assert check(output) == 0, "exactly 100 MiB → busy"


def test_just_below_threshold():
    output = "GPU-abc, 9999, some-process, 99\n"
    assert check(output) == 4, "99 MiB → noise, chips free"


def test_nvidia_smi_failure():
    from theseus.dispatch.solve import _check_plain_host_availability

    with patch(
        "theseus.dispatch.ssh.run",
        return_value=make_run_result("", ok=False),
    ):
        result = _check_plain_host_availability("fake-host", 4, timeout=5.0)
    assert result == 0, "nvidia-smi failure → unavailable"


if __name__ == "__main__":
    tests = [
        test_no_processes,
        test_xorg_only,
        test_zero_memory,
        test_na_memory,
        test_missing_memory_field,
        test_real_training_job,
        test_mixed_noise_and_real,
        test_exactly_at_threshold,
        test_just_below_threshold,
        test_nvidia_smi_failure,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
