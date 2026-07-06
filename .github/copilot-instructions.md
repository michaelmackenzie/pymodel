# Pymodel Repository Instruction Set (Unified Backend Driver)

This codebase is a standalone CLI framework (`pymodel`) orchestrating three strict statistical analysis backends:
1. `hfmodel` (pyhf-based execution loops)
2. `zmodel` (zfit and hepstats-based loops)
3. `roomodel` (pyROOT / RooFit workspace bindings)

## Architectural Constraints
- **Shared Abstraction Layer:** All backends inherit from the core interfaces in `python/backends/base.py`, `common.py`, and `card_parser.py`. When adding options or modifying entry points, respect the shared workflow abstractions.
- **Strict Backend Separation:** Do not mix syntax or paradigms between adapters. `zmodel` uses unbinned/binned numerical optimization graphs; `hfmodel` uses HistFactory JSON payloads and array tensors; `roomodel` interfaces with C++ ROOT bindings via pyROOT.
- **Context Injection:** Always read `ai_repo_skeleton.txt` in the root folder before recommending changes to ensure method signatures match the common driver interface.
- **Format Conventions:** Do not propose arbitrary file formatting. Models are saved distinctly (`.json` for pyhf, `.pkl` for zfit, `.root` for roomodel).

## Environment & Runtime Rules
- **Environment Initialization:** The Python path, repository command wrappers, and ROOT hooks are managed strictly via an environment setup script.
- **Terminal Execution:** Whenever writing or suggesting commands for a new terminal session, always instruct the user to run `source setup_env.sh` first to activate the workspace and Python environment.
