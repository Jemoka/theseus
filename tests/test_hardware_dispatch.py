"""Tests for Cluster-field plumbing through dispatch serialization and
bootstrap reconstruction.

`theseus/dispatch/bootstrap.py` is a template (contains ``__PRELOAD_IMPORTS__``
and other placeholders filled by `_generate_bootstrap`), so we can't import it
directly. We inline-load it here the same way ``_generate_bootstrap`` does:
read the text, substitute the placeholders with inert values, then exec.
"""

import json
from pathlib import Path

import pytest


BOOTSTRAP_TEMPLATE = Path(__file__).parent.parent / "theseus/dispatch/bootstrap.py"


@pytest.fixture
def bootstrap_ns(monkeypatch):
    """Load bootstrap.py into an isolated namespace with placeholders stubbed.

    Mirrors `_generate_bootstrap` (dispatch.py) — substitutes the template
    macros with empty/inert values and execs the resulting module. Uses the
    real `jax` (needed for flax imports that happen via `theseus.config`) but
    monkeypatches `process_count` to a deterministic value.
    """
    import jax

    monkeypatch.setattr(jax, "process_count", lambda: 1)

    src = BOOTSTRAP_TEMPLATE.read_text()
    # Same substitutions dispatch._generate_bootstrap makes, with inert values.
    src = src.replace("__PRELOAD_IMPORTS__", "# (preload stub)")
    src = src.replace("__CONFIG_YAML__", "")
    src = src.replace("__HARDWARE_JSON__", "{}")
    src = src.replace("__JOB_NAME__", "_test")
    src = src.replace("__PROJECT__", "")
    src = src.replace("__GROUP__", "")
    src = src.replace("__RESTORE_PATH__", "")

    ns: dict = {"__name__": "_bootstrap_under_test", "__file__": str(BOOTSTRAP_TEMPLATE)}
    exec(compile(src, str(BOOTSTRAP_TEMPLATE), "exec"), ns)
    return ns


@pytest.fixture
def make_hardware(tmp_path):
    """Build a minimal HardwareResult with all four overridable dirs set."""
    from theseus.base.chip import SUPPORTED_CHIPS
    from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult

    root = tmp_path / "root"
    root.mkdir()
    chip = SUPPORTED_CHIPS["h100"]
    cluster = Cluster(
        name="test",
        root=str(root),
        work=str(tmp_path / "work"),
        log=str(tmp_path / "log"),
        data=str(tmp_path / "data"),
        checkpoints=str(tmp_path / "ckpt"),
        results=str(tmp_path / "res"),
        status=str(tmp_path / "st"),
    )
    machine = ClusterMachine(name="host0", cluster=cluster, resources={chip: 4})
    return HardwareResult(chip=chip, hosts=[machine], total_chips=4)


class TestClusterDerivedDirs:
    def test_defaults_derive_from_root(self, tmp_path):
        from theseus.base.hardware import Cluster

        root = tmp_path / "r"
        root.mkdir()
        c = Cluster(name="x", root=str(root), work=str(tmp_path / "w"))
        assert c.data_dir == root / "data"
        assert c.checkpoints_dir == root / "checkpoints"
        assert c.results_dir == root / "results"
        assert c.status_dir == root / "status"
        # Each call auto-creates the dir.
        for d in (c.data_dir, c.checkpoints_dir, c.results_dir, c.status_dir):
            assert d.is_dir()

    def test_overrides_honored(self, tmp_path):
        from theseus.base.hardware import Cluster

        root = tmp_path / "r"
        root.mkdir()
        over = {
            "data": tmp_path / "od",
            "checkpoints": tmp_path / "ock",
            "results": tmp_path / "ore",
            "status": tmp_path / "os",
        }
        c = Cluster(
            name="x",
            root=str(root),
            work=str(tmp_path / "w"),
            data=str(over["data"]),
            checkpoints=str(over["checkpoints"]),
            results=str(over["results"]),
            status=str(over["status"]),
        )
        assert c.data_dir == over["data"]
        assert c.checkpoints_dir == over["checkpoints"]
        assert c.results_dir == over["results"]
        assert c.status_dir == over["status"]


class TestDispatchConfigPlumbing:
    def test_cluster_config_fields_default_none(self):
        from theseus.dispatch.config import ClusterConfig

        cfg = ClusterConfig(root="/a", work="/b")
        assert cfg.data is None
        assert cfg.checkpoints is None
        assert cfg.results is None
        assert cfg.status is None

    def test_parse_reads_overrides_from_yaml(self):
        from omegaconf import OmegaConf

        from theseus.dispatch.config import parse_dispatch_config

        yaml = OmegaConf.create(
            {
                "clusters": {
                    "c1": {
                        "root": "/r",
                        "work": "/w",
                        "data": "/d",
                        "checkpoints": "/ck",
                        "results": "/re",
                        "status": "/st",
                    }
                }
            }
        )
        parsed = parse_dispatch_config(yaml)
        c = parsed.clusters["c1"]
        assert c.data == "/d"
        assert c.checkpoints == "/ck"
        assert c.results == "/re"
        assert c.status == "/st"

    def test_get_cluster_threads_fields_through(self, tmp_path):
        from theseus.dispatch.config import (
            ClusterConfig,
            DispatchConfig,
            RemoteInventory,
        )

        root = tmp_path / "r"
        root.mkdir()
        cfg = DispatchConfig(
            clusters={
                "c1": ClusterConfig(
                    root=str(root),
                    work=str(tmp_path / "w"),
                    data="/d",
                    checkpoints="/ck",
                    results="/re",
                    status="/st",
                )
            }
        )
        cluster = RemoteInventory(cfg).get_cluster("c1")
        assert cluster.data == "/d"
        assert cluster.checkpoints == "/ck"
        assert cluster.results == "/re"
        assert cluster.status == "/st"


class TestSerializeRoundTrip:
    def test_serialize_includes_new_keys(self, make_hardware):
        from theseus.dispatch.dispatch import _serialize_hardware

        data = json.loads(_serialize_hardware(make_hardware))
        cluster_keys = data["hosts"][0]["cluster"]
        assert cluster_keys["data"].endswith("/data")
        assert cluster_keys["checkpoints"].endswith("/ckpt")
        assert cluster_keys["results"].endswith("/res")
        assert cluster_keys["status"].endswith("/st")

    def test_reconstruct_round_trip(self, make_hardware, bootstrap_ns):
        from theseus.dispatch.dispatch import _serialize_hardware

        data = json.loads(_serialize_hardware(make_hardware))
        recon = bootstrap_ns["_reconstruct_hardware"](data)
        c = recon.hosts[0].cluster
        expected = make_hardware.hosts[0].cluster
        assert c.data == expected.data
        assert c.checkpoints == expected.checkpoints
        assert c.results == expected.results
        assert c.status == expected.status

    def test_env_overrides_win(self, make_hardware, bootstrap_ns, monkeypatch):
        from theseus.dispatch.dispatch import _serialize_hardware

        monkeypatch.setenv("THESEUS_DISPATCH_DATA_OVERRIDE", "/env/data")
        monkeypatch.setenv("THESEUS_DISPATCH_CHECKPOINTS_OVERRIDE", "/env/ck")
        monkeypatch.setenv("THESEUS_DISPATCH_RESULTS_OVERRIDE", "/env/re")
        monkeypatch.setenv("THESEUS_DISPATCH_STATUS_OVERRIDE", "/env/st")

        data = json.loads(_serialize_hardware(make_hardware))
        recon = bootstrap_ns["_reconstruct_hardware"](data)
        c = recon.hosts[0].cluster
        assert c.data == "/env/data"
        assert c.checkpoints == "/env/ck"
        assert c.results == "/env/re"
        assert c.status == "/env/st"

    def test_multihost_expansion_shares_cluster(
        self, make_hardware, bootstrap_ns, monkeypatch
    ):
        """When jax.process_count > len(hosts), bootstrap expands the hosts
        list and every expanded ClusterMachine shares the same Cluster object,
        so the 4 new fields propagate to every host automatically."""
        import jax

        from theseus.dispatch.dispatch import _serialize_hardware

        monkeypatch.setattr(jax, "process_count", lambda: 3)

        data = json.loads(_serialize_hardware(make_hardware))
        recon = bootstrap_ns["_reconstruct_hardware"](data)
        assert len(recon.hosts) == 3
        for h in recon.hosts:
            assert h.cluster.data == make_hardware.hosts[0].cluster.data
            assert h.cluster.checkpoints == make_hardware.hosts[0].cluster.checkpoints
            assert h.cluster.results == make_hardware.hosts[0].cluster.results
            assert h.cluster.status == make_hardware.hosts[0].cluster.status
