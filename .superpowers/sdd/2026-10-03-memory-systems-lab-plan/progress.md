# SDD ledger — plan: docs/superpowers/plans/2026-10-03-memory-systems-lab-plan.md

Ruling: Work in the current workspace without creating a git worktree — the user explicitly prohibited all Git operations, and code changes are authorized here; cost if wrong: the implementation shares the active checkout until the user reviews it.

Preflight: Read the approved spec and plan; all module paths and interfaces match the scaffold. Dataset hashes captured before implementation. Do not run Git commands or change Git metadata.

Task 1: complete (tests: `src/test_config.py` → 3/3 pass; initial run failed on the scaffold as expected).

Ruling: Add backward-compatible `state_dir` keyword to `load_config()` — the benchmark needs isolated temporary state while the scaffold's one-argument calls must continue to work; cost if wrong: a slightly wider config API.

Task 2: complete (tests: profile IO/edit, path safety, confident extraction/correction, pet-name preservation, compaction → 7/7 pass).

Task 3: complete (agent behavior tests: same-thread Baseline recall, cross-thread Baseline forgetting, Advanced cross-thread recall, correction recall, long-thread prompt reduction → 5/5 pass; original scaffold tests failed before implementation).

Ruling: Keep benchmark contract tests in dedicated `src/test_benchmark.py` — this keeps the agent behavior file focused while testing the benchmark's own boundaries; cost if wrong: the requested `pytest src/test_agents.py -v` command alone does not execute these extra tests, so the complete local suite will also run.

Task 4: complete (benchmark unit tests: 7/7 pass; benchmark command: exit 0 twice with byte-for-byte identical output; both datasets show Advanced recall 1.000; stress shows 11,001 vs 23,273 prompt tokens and 4 compactions).

Task 5: complete (README and `STEP8.md` include reproducible output/analysis; `python src/benchmark.py` exited 0 twice with matching tables; all 32 tests pass; both dataset SHA-256 values match the pre-work values; no `.env` or repository `state/` found).

Final review: complete. Review identified a location overwrite from hypothetical/uncertain claims, a missed correction after a historical clause, profile path collisions, and missing edge-case assertions. Added failing tests, fixed the extractor and profile-path identity, and reran the full suite and benchmark. Review notes about bounded summaries and malformed turns also now have test coverage. No Git commands or Git metadata changes were used.
