"""Tests for _check_plain_host_availability GPU memory parsing logic.

Migrated from scripts/test_gpu_availability.py.
"""

from unittest.mock import MagicMock, patch
import pytest


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


class TestGPUAvailability:
    def test_no_processes(self):
        assert check("") == 4

    def test_xorg_only(self):
        output = "GPU-abc, 3374, /usr/lib/xorg/Xorg, 4\nGPU-def, 3374, /usr/lib/xorg/Xorg, 4\n"
        assert check(output) == 4

    def test_zero_memory(self):
        output = "GPU-abc, 1234, some-daemon, 0\n"
        assert check(output) == 4

    def test_na_memory(self):
        output = "GPU-abc, 1234, some-process, N/A\n"
        assert check(output) == 4

    def test_missing_memory_field(self):
        output = "GPU-abc, 1234, some-process\n"
        assert check(output) == 4

    def test_real_training_job(self):
        output = "GPU-abc, 9999, python, 38000\n"
        assert check(output) == 0

    def test_mixed_noise_and_real(self):
        output = "GPU-abc, 3374, /usr/lib/xorg/Xorg, 4\nGPU-def, 9999, python, 38000\n"
        assert check(output) == 0

    def test_exactly_at_threshold(self):
        output = "GPU-abc, 9999, python, 100\n"
        assert check(output) == 0

    def test_just_below_threshold(self):
        output = "GPU-abc, 9999, some-process, 99\n"
        assert check(output) == 4

    def test_nvidia_smi_failure(self):
        from theseus.dispatch.solve import _check_plain_host_availability

        with patch(
            "theseus.dispatch.ssh.run",
            return_value=make_run_result("", ok=False),
        ):
            result = _check_plain_host_availability("fake-host", 4, timeout=5.0)
        assert result == 0
