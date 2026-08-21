<!-- What changes, and why. Link the issue if one exists. -->

## Eval evidence

<!-- The two rules from CONTRIBUTING.md:

     1. Every quality-affecting change is eval-gated. If this PR can
        alter what gets retrieved or generated, paste the eval result
        (smoke set for regressions, large set for design verdicts).

     2. Never change the golden set and the system under test in the
        same measured comparison. If this PR edits eval data, re-record
        baselines in this same PR and say why.

     Not quality-affecting? Say so in one line and delete the rest. -->

## Checklist

- [ ] `pytest tests/` passes
- [ ] `black --check src/ tests/ && isort --check-only src/ tests/ && flake8 src/ tests/`
- [ ] `mypy src/ragstone` clean
- [ ] CHANGELOG.md updated for user-visible changes
