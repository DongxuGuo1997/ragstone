"""
Bench-generator invariants that must survive any refactor.

The committed 40-person bench is a measurement artifact: two baseline
keys and a data_sha fingerprint stand on it. The --scale parameter must
therefore be provably inert at scale=1 (byte-identical specs), and the
XL population must preserve the bench's engineered properties — above
all that a08 stays unsatisfiable at any population size.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(EVALS_DIR))
    spec = importlib.util.spec_from_file_location(
        "generate_staffing", EVALS_DIR / "generate_staffing.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestScaleInvariants:
    def test_scale_one_matches_the_committed_specs_exactly(self, bench):
        built = bench.build_personas(scale=1)
        committed = json.loads(
            (EVALS_DIR / "staffing_personas.json").read_text(encoding="utf-8")
        )
        assert built == committed

    def test_xl_population_counts_ids_and_unique_names(self, bench):
        personas = bench.build_personas(scale=3)
        assert len(personas) == 120
        assert personas[0]["id"] == "cv001"  # three-digit ids in XL
        names = [p["name"] for p in personas]
        assert len(set(names)) == len(names)

    def test_a08_stays_unsatisfiable_at_scale(self, bench):
        personas = bench.build_personas(scale=5)
        a08 = next(a for a in bench.ASSIGNMENTS if a["id"] == "a08")
        strong, partial = bench.expected_tiers(personas, a08)
        assert strong == []
        assert len(partial) >= 2

    def test_strong_pools_grow_with_the_population(self, bench):
        small = bench.build_personas(scale=1)
        big = bench.build_personas(scale=5)
        a01 = next(a for a in bench.ASSIGNMENTS if a["id"] == "a01")
        strong_small, _ = bench.expected_tiers(small, a01)
        strong_big, _ = bench.expected_tiers(big, a01)
        assert len(strong_big) > len(strong_small)
