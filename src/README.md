# Day 17 Memory Systems Lab

This folder contains the completed lab implementation.

- `config.py` and `model_provider.py`: offline defaults and optional adapters for OpenAI, custom OpenAI-compatible endpoints, Gemini, Anthropic, Ollama, and OpenRouter.
- `memory_store.py`: safe `User.md` profiles, conservative fact extraction, correction upserts, token estimation, and bounded thread compaction.
- `agent_baseline.py`: thread-only memory.
- `agent_advanced.py`: thread memory, persistent user profile, and compact memory.
- `benchmark.py`: deterministic Standard and Long-Context Stress comparisons.
- `test_agents.py`: isolated behavior tests for profile operations, extraction, corrections, cross-session recall, and compaction.
- `test_benchmark.py` and `test_config.py`: benchmark and configuration contract tests.

Run commands from the repository root. The offline benchmark does not need provider SDKs or credentials:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Run the full lab suite, including configuration and benchmark validation, with:

```bash
pytest src -v
```

Set `LLM_PROVIDER` and its matching API key to opt into a live provider in an agent application. The benchmark deliberately forces offline mode so its results stay repeatable.
