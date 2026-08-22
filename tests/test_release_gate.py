from pathlib import Path

from scripts.run_release_gate import run_release_gate


ROOT = Path(__file__).parents[1]


def test_offline_release_gate_passes_every_suite() -> None:
    report = run_release_gate(fixtures_path=ROOT / "data/golden/fixtures.json")

    assert report["passed_all"] is True
    assert report["total"] == 93
    assert report["passed"] == 93
    assert [suite["name"] for suite in report["suites"]] == [
        "synthetic_baseline",
        "bank_transfer_not_received",
        "cashback_not_received",
        "google_play_refund",
        "workflow_boundaries",
    ]
