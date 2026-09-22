# Archive

Pre-service experiments, kept because the service does not reproduce them. Nothing here is
imported by `src/so1/` or the tests.

| file | what it is |
|---|---|
| `compare_models.py` | Two-endpoint harness (E4B vs 26B-A4B). Compares the logprob readout against ordinary generation, and includes the rules-decomposition experiment (one broad eligibility question vs six yes/no sub-checks combined in code). The service reproduces neither comparison. |
| `compare_20260922_004749.json` / `.csv` | Results of that run. |
| `longest.py` | Long-context probe and the original 15-case run. Its case generator is vendored at `tests/fixtures/eval_cases.py`; the rest is superseded by the service and `scripts/eval_service.py`. |
| `results_20260922_000945.json` | The 14/15 baseline from `longest.py`, vendored at `tests/fixtures/baseline_e4b_direct.json`. |
