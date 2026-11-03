# CHAT_CONTEXT: what came from the claude.ai chats

Source: claude.ai export (2026-11-03), conversation "MBI reliability report review" (2026-10-27) and the project "RRS-MBI". This is background only. The repo files (`CLAUDE.md`, `RESEARCH_CONTEXT.md`, `RRS_code_audit.md`) and the results in `results_audit/` are the source of truth. If anything here disagrees with them, they win.

## How the work started
1. rr asked for an examiner-style review of the internship report (Laporan Kerja Praktik). Rules: ignore front matter and numbering; keep the logical bridge from failure-mode characterization to RRS-MBI; every parameter must be justified by physics, literature or the report's own data. Output: three sections, most to least severe.
2. rr asked for a corrected PDF. Fixes are highlighted yellow. Open questions are highlighted red, written like `[add parameter for dice score here]`.
3. rr said the foundational literature justification is an author choice based on an ablation study that is not in the report. It stays out of scope.
4. The repo (branch `fix/preprocessing`) was then audited. That produced `RRS_code_audit.md`, `fixes.patch` and `sim.py`. The data is too large to upload, so the work moved to Claude Code.

## Examiner findings that matter for the report
- The baseline error (31.84 mm) is about the same as always guessing the centre. Later confirmed in `results_audit/final/pipeline_vs_trivial.txt`.
- Figure 5 labels the antenna ring R = 50 mm. The real Gen-2 ring is about 20-23 cm. Later confirmed (bug 2).
- The "boundary signal collapse" explanation has the physics reversed: boundary tumours are closer to the antennas, not farther.
- SVM and U-Net failures do not prove a physical limit. The U-Net showed class-imbalance and double-sigmoid signs. Later confirmed (bug 6).
- The report claims significance but its own permutation p = 0.098 contradicts it.
- Citations to fix: coherence factor should cite Mallart & Fink, Li & Li, or Klemm, not Fear 2002.
- Well supported: dataset facts, ICZT settings, early-time gating rationale.
- Needs sensitivity checks or more justification: Tukey windows, per-channel normalization, "Hybrid TVSVD" thresholds.

## Open items (not yet done)
- Corrected report PDF with yellow/red marks, rebuilt on the audit's final numbers.
- Decide whether "the limit is physical, not algorithmic" can stay (see `results_audit/final/`).
- The conversation also mentioned a GIBR/bent-ray attempt. No such code is in the repo. Do not recreate it.
