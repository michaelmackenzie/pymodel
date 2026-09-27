---
name: compare-combine
description: Check whether pymodel reproduces Combine on a given datacard (asymptotic limits, best fit, 68% interval) for every backend, without touching the original card's directory. Use when asked whether pymodel agrees with Combine on a card, or when debugging a discrepancy.
---

# Compare pymodel with Combine on a datacard

1. Use the Combine environment (see the validate-stats skill, step 1).
2. `cd` to the directory that the card's shape paths are relative to, which is where the user
   runs Combine (for mumep_ana: `mumep_ana/analysis/combine`). Then run:
   ```bash
   python3 /exp/mu2e/app/users/mmackenz/mumep/pymodel/scripts/compare_combine.py datacards/<card>.txt \
       --workdir /tmp/<scratch>/cmp_<name> [--backends roomodel,zmodel,hfmodel] [--rmax 20]
   ```
   The script copies the card and its shape files into the work directory, and runs Combine
   and pymodel only there. It never writes next to the original card.
3. Read the table. Limits within about 0.3% and a matching 68% upper edge mean agreement. A
   backend that refuses the card prints `UnsupportedByBackend` with the missing features.
   That is a support gap, not a discrepancy.
4. To localise a real discrepancy:
   - Run `pymodel <backend> inspect <card>` and compare the per-process expected yields with
     Combine's `n_exp_final_bin<ch>_proc_<p>` functions in `cmp_ws.root`.
   - Compare NLL differences between parameter points with
     `pymodel <backend> nll <card> --at r=1,theta=0.5`. Compare differences, not absolute
     values, because Combine's NLL has other constants.
   - Check `tests/backend_roomodel_check.py`, which contains a worked Combine-NLL comparison.
5. If `--rmax` was too small (the best fit sits at the bound), rerun with a larger value.
   Both Combine and pymodel flag this.

Never run Combine inside `mumep_ana` or the repository. Keep each run within a few minutes
of CPU.
