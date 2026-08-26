#!/bin/bash
# ===========================================================================
# bench_full.sh — full benchmark pipeline (relayout -> fast_run x3 -> metrics -> report)
# ===========================================================================
# For the main session to run in the background. This script makes no
# numerical decisions itself — it just runs five already-validated steps in
# order, times them, and rolls the results up into a markdown report.
#
#   1 relayout   114 GB OME-TIFF (ZTCYX)  ->  60 GiB T-major .npy
#   2 fast_run   .npy + float64 + replicate   --compare against reportB's serial output
#   3 fast_run   .npy + float32 + replicate   compute max|diff| / Pearson r against step 2
#   4 fast_run   .npy + float64 + improved
#   5 cpstab.metrics  truth vs step 2 vs step 4        ->  bench_report.md
#
# Usage
# -----
#   bash scripts/bench_full.sh                  # full run, start to finish
#   bash scripts/bench_full.sh --steps 2,3,4,5  # only run the given steps (comma-separated)
#   bash scripts/bench_full.sh --from 3         # start from step 3
#   bash scripts/bench_full.sh --dry-run        # only print commands, don't execute
#   bash scripts/bench_full.sh --subset         # run the whole pipeline on the 40-frame subset (rehearsal)
#   WORKERS=14 bash scripts/bench_full.sh       # override worker count (default 10)
#
# Idempotent: each step checks both whether the artifact exists **and**
# whether the code fingerprint matches — only skips if both hold
# (--force forces a rerun; combine with --steps to force just one step). So
# if it dies partway through, rerunning the same command just works — it
# won't redo finished steps; and after changing cpstab/fast_run, rerunning
# will redo the affected steps on its own rather than passing off a stale
# artifact as a fresh result (see DESIGN NOTES 1). Step 1's relayout has its
# own .partial + integrity invariant built in, so a half-written file never
# appears under its final name (scripts/relayout.py DESIGN NOTES 5/8).
#
# All stdout/stderr appends to $WS/bench_log.txt; each step is also timed
# with /usr/bin/time -p, and the timing line also lands in
# $BENCH/_timings.tsv for step 5 to build its table.
# ===========================================================================
set -u

# ---- fixed paths (match prior validation — don't casually change) --------------------------------
PY="${PY:-python}"
PYPORT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CPSTAB_WORKSPACE:?set CPSTAB_WORKSPACE to the directory holding the run's data}"

RAW_TIF="$WS/FAD-F_1_raw.ome.tif"
RAW_NPY="$WS/FAD-F_1_raw.tzcyx.npy"
REF_ZPROJ="$WS/reportB/port_run/FAD-F_1_raw.ome_mean_zproj.tif"
TRUTH="$WS/truth/20230201-FAD-F_230620_001_zproj.tif"
STEM="FAD-F_1_raw.tzcyx"          # relayout artifact, minus its final extension
WITH_TRUTH=1

LOG="$WS/bench_log.txt"
REPORT="$WS/bench_report.md"
BENCH="$WS/bench"

WORKERS="${WORKERS:-10}"
REFCHANNEL="${REFCHANNEL:-1}"
SCALE="${SCALE:-4}"
CHUNKSIZE="${CHUNKSIZE:-20}"
PROJ_RANGE="${PROJ_RANGE:-quarter}"
READ_MB="${READ_MB:-2048}"
VERIFY="${VERIFY:-16}"
METRIC_STRIDE="${METRIC_STRIDE:-8}"

STEPS="1,2,3,4,5"
DRY=0
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --steps)   STEPS="$2"; shift 2 ;;
    --from)    STEPS=$(echo "1 2 3 4 5" | tr ' ' '\n' | awk -v n="$2" '$1>=n' | paste -sd, -); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --force)   FORCE=1; shift ;;
    --subset)
      # 40-frame rehearsal: same pipeline, swap in the subset's input and reference output.
      RAW_TIF="$WS/FAD-F_1_T0-39.tif"
      RAW_NPY="$WS/FAD-F_1_T0-39.tzcyx.npy"
      REF_ZPROJ="$WS/reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif"
      STEM="FAD-F_1_T0-39.tzcyx"
      # The subset has no matching MATLAB truth (truth is the full 1500-frame
      # run) — shapes don't line up, and putting it in the same metrics
      # table would just mislead -> subset mode skips the truth comparison.
      WITH_TRUTH=0
      LOG="$WS/bench_log_subset.txt"
      REPORT="$WS/bench_report_subset.md"
      BENCH="$WS/bench_subset"
      METRIC_STRIDE=1
      shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

OUT_F64="$BENCH/f64_replicate"
OUT_F32="$BENCH/f32_replicate"
OUT_IMP="$BENCH/f64_improved"
ZP_F64="$OUT_F64/${STEM}_mean_zproj.tif"
ZP_F32="$OUT_F32/${STEM}_mean_zproj.tif"
ZP_IMP="$OUT_IMP/${STEM}_mean_zproj.tif"

TIMINGS="$BENCH/_timings.tsv"
CMP2="$BENCH/_step2_bitwise.txt"
DIFF3="$BENCH/_step3_f32_vs_f64.txt"
MET5="$BENCH/_step5_metrics.md"

mkdir -p "$BENCH" "$OUT_F64" "$OUT_F32" "$OUT_IMP"

# ---- helpers ---------------------------------------------------------------
say() { echo "$@" | tee -a "$LOG"; }

want() { echo ",$STEPS," | grep -q ",$1,"; }

# Did this step **actually run in this invocation** (as opposed to being
# excluded by --steps / skipped because the artifact already exists)?
# Postprocessing only trusts this: e.g. step 2's comparison verdict is
# grepped out of the log, and the log is append-only — if the step was
# skipped, the grep would pick up the line from a previous run (possibly
# for different data).
RAN=""
ran() { echo ",$RAN," | grep -q ",$1,"; }
RAN3B=0

mtime_of() { stat -f '%Sm' -t '%F %T' "$1" 2>/dev/null || echo '?'; }

# ---- code fingerprint: run_step's second rerun criterion -----------------------------------
# Checking only "does the artifact exist" means: if the code changed but the
# artifact was written by the old code, this step gets silently skipped and
# the stale artifact gets republished under today's date. This isn't
# hypothetical: step 4's f64_improved tif was 73 minutes older than
# improved.py, and rerunning the exact same command at the time produced
# output that differed in 95.7% of pixels (max|diff| 202 counts) — the
# "improved" score in the report simply wasn't from the current code. So
# each step also stores a sha256 of "the code files it depends on + this
# invocation's command line" alongside the artifact.
STAMP_FILES=()      # code this step depends on (set before each run_step call)
STAMP_ADVISORY=0    # 1 = fingerprint mismatch only warns, doesn't auto-rerun (for steps too costly to rerun automatically)

step_sig() {
  { printf '%s\n' "$@"
    for f in ${STAMP_FILES[@]+"${STAMP_FILES[@]}"}; do
      [ -e "$f" ] && shasum -a 256 "$f"
    done
  } | shasum -a 256 | awk '{print $1}'
}

# Old artifacts with no fingerprint (produced before this refactor) fall
# back to mtime comparison, so step 1's 60 GiB conversion doesn't get judged
# stale right out of the gate. Lists code files newer than the artifact;
# empty = artifact is trustworthy.
newer_code() {
  local art="$1" f
  for f in ${STAMP_FILES[@]+"${STAMP_FILES[@]}"}; do
    if [ -e "$f" ] && [ "$f" -nt "$art" ]; then basename "$f"; fi
  done
}

# run_step <number> <name> <artifact path|-> <command...>
run_step() {
  local n="$1" name="$2" artifact="$3"; shift 3
  if ! want "$n"; then say "[step $n] $name — SKIP (not in --steps)"; return 0; fi
  local stamp="" sig="" old="" stale=""
  if [ "$artifact" != "-" ]; then
    stamp="$BENCH/_stamp_$n.sha"
    sig=$(step_sig "$@")
  fi
  if [ "$artifact" != "-" ] && [ -e "$artifact" ] && [ "$FORCE" -eq 0 ]; then
    [ -e "$stamp" ] && old=$(cat "$stamp")
    if [ -z "$old" ]; then
      stale=$(newer_code "$artifact" | tr '\n' ' ')
      if [ -z "$stale" ]; then
        printf '%s\n' "$sig" > "$stamp"
        say "[step $n] $name — SKIP (artifact exists and is newer than all dependent code; fingerprint backfilled)"
        return 0
      fi
      stale="code newer than the artifact: $stale"
    elif [ "$old" = "$sig" ]; then
      say "[step $n] $name — SKIP (artifact exists, code+params fingerprint matches: $artifact)"
      return 0
    else
      stale="fingerprint mismatch (artifact ${old:0:12}... vs current ${sig:0:12}...)"
    fi
    if [ "$STAMP_ADVISORY" -eq 1 ]; then
      say "[step $n] $name — SKIP (artifact exists)"
      say "  WARNING: $stale"
      say "  WARNING: rerunning this step is too expensive to do automatically. To force a redo: --force --steps $n"
      return 0
    fi
    say "[step $n] $name — artifact stale ($stale) -> rerunning"
  fi
  say ""
  say "==========================================================="
  say "[step $n] $name"
  say "  started $(date '+%F %T')"
  say "  cmd: $*"
  say "==========================================================="
  if [ "$DRY" -eq 1 ]; then say "  (dry-run, not executed)"; return 0; fi
  local t0 t1 rc
  t0=$(date +%s)
  # /usr/bin/time -p writes to stderr; after 2>&1 it all goes through tee, so
  # real/user/sys land in the log too
  /usr/bin/time -p "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$n" "$name" "$((t1 - t0))" "$rc" >> "$TIMINGS"
  RAN="$RAN,$n"
  say "  [step $n] rc=$rc  wall=$((t1 - t0))s ($(awk -v s=$((t1 - t0)) 'BEGIN{printf "%.1f", s/60}') min)"
  if [ "$rc" -ne 0 ]; then
    say "  [step $n] FAILED — later steps depend on it, aborting."
    exit "$rc"
  fi
  # Fingerprint is only recorded on success; a failed/half-written artifact
  # will never be treated as complete by a later invocation.
  [ -n "$stamp" ] && printf '%s\n' "$sig" > "$stamp"
  return 0
}

# ---- opening -----------------------------------------------------------
say ""
say "###########################################################"
say "# bench_full.sh  $(date '+%F %T')"
say "#   raw       : $RAW_TIF"
say "#   store     : $RAW_NPY"
say "#   reference : $REF_ZPROJ"
say "#   out       : $BENCH"
say "#   workers=$WORKERS refchannel=$REFCHANNEL scale=$SCALE"
say "#   chunksize=$CHUNKSIZE proj_range=$PROJ_RANGE read_mb=$READ_MB"
say "#   steps=$STEPS dry_run=$DRY force=$FORCE"
say "#   python    : $PY"
say "#   host      : $(hostname)  cpus=$(sysctl -n hw.ncpu 2>/dev/null || echo ?)"
say "#   free disk : $(df -h "$WS" | tail -1 | awk '{print $4}')"
say "###########################################################"

# ===========================================================================
# 1 relayout: ZTCYX OME-TIFF -> T-major (T, Z, C, Y, X) .npy
# ===========================================================================
# relayout itself also refuses to overwrite (default behavior), so --force
# must be threaded all the way through, or run_step won't skip while
# relayout reports "refusing to overwrite" — a false failure that only
# shows up under --force.
RELAYOUT_ARGS=(--in "$RAW_TIF" --out "$RAW_NPY" --verify "$VERIFY")
[ "$FORCE" -eq 1 ] && RELAYOUT_ARGS+=(--force)
# Step 1's fingerprint is ADVISORY: rerunning it means overwriting 60 GiB,
# and relayout itself refuses to overwrite anyway — auto-rerunning would
# just turn into a false failure or an unwanted conversion nobody asked
# for. A fingerprint mismatch just gets flagged loudly, and a human decides.
STAMP_FILES=("$PYPORT/scripts/relayout.py")
STAMP_ADVISORY=1
run_step 1 "relayout -> T-major .npy" "$RAW_NPY" \
  "$PY" "$PYPORT/scripts/relayout.py" "${RELAYOUT_ARGS[@]}"
STAMP_ADVISORY=0

# Steps 2, 3, 4 all use fast_run + the whole cpstab package; if any .py
# changes, the artifact no longer represents current code.
PIPE_CODE=("$PYPORT/fast_run.py" "$PYPORT"/cpstab/*.py)

# ===========================================================================
# 2 float64 + replicate — bitwise comparison against reportB's serial output
# ===========================================================================
STAMP_FILES=("${PIPE_CODE[@]}")
run_step 2 "fast_run npy float64 replicate (+bitwise vs reference)" "$ZP_F64" \
  "$PY" "$PYPORT/fast_run.py" \
    --raw "$RAW_NPY" --out-dir "$OUT_F64" --workers "$WORKERS" \
    --refchannel "$REFCHANNEL" --scale "$SCALE" --chunksize "$CHUNKSIZE" \
    --proj-range "$PROJ_RANGE" --dtype float64 --mode replicate \
    --read-mb "$READ_MB" --compare "$REF_ZPROJ"

if ran 2; then
  grep -a "bitwise identical" "$LOG" | tail -1 > "$CMP2"
  say "  [step 2] $(cat "$CMP2")"
fi

# ===========================================================================
# 3 float32 + replicate — timing + max|diff| / Pearson r against step 2's output
# ===========================================================================
STAMP_FILES=("${PIPE_CODE[@]}")
run_step 3 "fast_run npy float32 replicate" "$ZP_F32" \
  "$PY" "$PYPORT/fast_run.py" \
    --raw "$RAW_NPY" --out-dir "$OUT_F32" --workers "$WORKERS" \
    --refchannel "$REFCHANNEL" --scale "$SCALE" --chunksize "$CHUNKSIZE" \
    --proj-range "$PROJ_RANGE" --dtype float32 --mode replicate \
    --read-mb "$READ_MB"

if want 3 && [ "$DRY" -eq 0 ] && [ -e "$ZP_F32" ] && [ -e "$ZP_F64" ]; then
  say ""
  say "[step 3b] float32 vs float64 per-frame diff (streaming, no full load)"
  # Two 1.5 GB uint16 TIFFs: accumulate Pearson's six sums frame by frame,
  # keeping only max|diff| and the counts.
  "$PY" - "$ZP_F32" "$ZP_F64" <<'PY' 2>&1 | tee -a "$LOG" | tee "$DIFF3"
import sys
import numpy as np
import tifffile

fa, fb = sys.argv[1], sys.argv[2]


def _open(p):
    """memmap when the TIFF is contiguous, else a plain read."""
    try:
        return np.asarray(tifffile.memmap(p))
    except (ValueError, NotImplementedError):
        return np.asarray(tifffile.imread(p))


A, B = _open(fa), _open(fb)
assert A.shape == B.shape, "shape %r != %r" % (A.shape, B.shape)
n = A.shape[0]
sa = sb = saa = sbb = sab = 0.0
cnt = 0
mx = 0
tot = 0
for i in range(n):                       # one page at a time, constant memory
    a = np.asarray(A[i], dtype=np.float64).ravel()
    b = np.asarray(B[i], dtype=np.float64).ravel()
    d = a - b
    m = np.abs(d).max() if d.size else 0.0
    mx = max(mx, float(m))
    cnt += int(np.count_nonzero(d))
    tot += d.size
    sa += a.sum(); sb += b.sum()
    saa += (a * a).sum(); sbb += (b * b).sum(); sab += (a * b).sum()
cov = sab - sa * sb / tot
va = saa - sa * sa / tot
vb = sbb - sb * sb / tot
r = cov / np.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
print("shape            : %r" % (A.shape,))
print("bitwise identical: %s" % (cnt == 0))
print("max|diff| (counts): %g" % mx)
print("differing pixels  : %d / %d (%.3e)" % (cnt, tot, cnt / float(tot)))
print("Pearson r         : %.15f" % r)
PY
  RAN3B=1
fi

# ===========================================================================
# 4 float64 + improved
# ===========================================================================
STAMP_FILES=("${PIPE_CODE[@]}")
run_step 4 "fast_run npy float64 improved" "$ZP_IMP" \
  "$PY" "$PYPORT/fast_run.py" \
    --raw "$RAW_NPY" --out-dir "$OUT_IMP" --workers "$WORKERS" \
    --refchannel "$REFCHANNEL" --scale "$SCALE" --chunksize "$CHUNKSIZE" \
    --proj-range "$PROJ_RANGE" --dtype float64 --mode improved \
    --read-mb "$READ_MB"

# ===========================================================================
# 5 cpstab.metrics: truth vs replicate vs improved -> bench_report.md
# ===========================================================================
if want 5 && [ "$DRY" -eq 1 ]; then
  say ""
  say "[step 5] cpstab.metrics + bench_report.md — (dry-run, not executed)"
  say "  cmd: $PY -m cpstab.metrics --stride $METRIC_STRIDE [truth=] [replicate_f64=] [improved_f64=] [replicate_f32=]"
  say "  report -> $REPORT"
fi

if want 5 && [ "$DRY" -eq 0 ]; then
  say ""
  say "==========================================================="
  say "[step 5] cpstab.metrics (stride=$METRIC_STRIDE)"
  say "==========================================================="
  MET_ARGS=""
  [ "$WITH_TRUTH" -eq 1 ] && [ -e "$TRUTH" ] && MET_ARGS="truth=$TRUTH"
  [ -e "$ZP_F64" ] && MET_ARGS="$MET_ARGS replicate_f64=$ZP_F64"
  [ -e "$ZP_IMP" ] && MET_ARGS="$MET_ARGS improved_f64=$ZP_IMP"
  [ -e "$ZP_F32" ] && MET_ARGS="$MET_ARGS replicate_f32=$ZP_F32"
  t0=$(date +%s)
  ( cd "$PYPORT" && /usr/bin/time -p "$PY" -m cpstab.metrics \
      --stride "$METRIC_STRIDE" $MET_ARGS ) 2>&1 | tee -a "$LOG" | \
      grep -a -v '^real\|^user\|^sys' > "$MET5"
  # The pipe's last stage is grep, so $? belongs to grep — must use
  # PIPESTATUS[0] here, or metrics raising an exception would still record
  # rc=0, and the traceback would get pasted into section 4 as if it were
  # the metrics table. Step 5 is the only step that doesn't go through
  # run_step, and that's exactly what let it slip past DESIGN NOTES 2 before.
  rc5=${PIPESTATUS[0]}
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' 5 "cpstab.metrics" "$((t1 - t0))" "$rc5" >> "$TIMINGS"
  RAN="$RAN,5"
  say "  [step 5] rc=$rc5  wall=$((t1 - t0))s"
  if [ "$rc5" -ne 0 ]; then
    say "  [step 5] FAILED — $MET5 contains a traceback, not a metrics table; report not written."
    say "$(cat "$MET5")"
    exit "$rc5"
  fi

  # ---- roll up into bench_report.md ------------------------------------------
  {
    echo "# Shipley stabilization pipeline — full benchmark (bench_full.sh)"
    echo
    echo "- generated: $(date '+%F %T')"
    echo "- machine: $(hostname), cpus=$(sysctl -n hw.ncpu 2>/dev/null || echo '?'), workers=$WORKERS"
    echo "- raw data: \`$RAW_TIF\`"
    echo "- T-major store: \`$RAW_NPY\`"
    echo "- params: refchannel=$REFCHANNEL scale=$SCALE chunksize=$CHUNKSIZE proj_range=$PROJ_RANGE read_mb=$READ_MB"
    echo "- full log: \`$LOG\`"
    echo
    echo "## 1. Per-step timings"
    echo
    echo "| # | step | wall (s) | wall (min) | rc | source |"
    echo "|---|---|---|---|---|---|"
    if [ -e "$TIMINGS" ]; then
      # _timings.tsv is append-only (resuming needs to keep the previous
      # round's completed-step timings), so the same step can have multiple
      # lines — take only the **last** one per step, sort by step number,
      # then add a total. The "source" column was added later: without it,
      # even a `--steps 5` run would print timings for steps 1-4 too, and it
      # would read exactly like a full run. The total only adds up steps
      # that actually ran this time.
      awk -F'\t' -v ran=",$RAN," '
        { t[$1] = $0 }
        END {
          tot = 0
          for (i = 1; i <= 9; i++) {
            if (i in t) {
              split(t[i], f, "\t")
              here = (index(ran, "," i ",") > 0)
              printf "| %s | %s | %s | %.1f | %s | %s |\n", f[1], f[2], f[3], \
                     f[3]/60, f[4], (here ? "this run" : "**previous (not run this time)**")
              if (here) tot += f[3]
            }
          }
          printf "| | **total (this run)** | **%d** | **%.1f** | | |\n", tot, tot/60
        }' "$TIMINGS"
    fi
    echo
    echo "## 2. Bitwise comparison (step 2 float64 replicate vs serial reference)"
    echo
    echo "reference: \`$REF_ZPROJ\`"
    echo
    echo '```'
    # Only "ran 2" is fresh. $CMP2 is left over from the previous run;
    # unconditionally catting it would republish the previous conclusion
    # (possibly for different data, a different reference, different code)
    # under today's date — while the old "(not run)" fallback branch only
    # triggers the very first time the file has never been generated, i.e.
    # it only worked on history's first run.
    if ran 2; then
      [ -e "$CMP2" ] && cat "$CMP2" || echo "(not run)"
    elif [ -e "$CMP2" ]; then
      echo "WARNING: step 2 wasn't run this time. Below is the previous conclusion, written $(mtime_of "$CMP2");"
      echo "WARNING: it may not match this report's params/reference path above, or the current code."
      cat "$CMP2"
    else
      echo "(not run)"
    fi
    echo '```'
    echo
    echo "## 3. float32 fast mode vs float64 (both replicate)"
    echo
    echo '```'
    if [ "$RAN3B" -eq 1 ]; then
      [ -e "$DIFF3" ] && cat "$DIFF3" || echo "(not run)"
    elif [ -e "$DIFF3" ]; then
      echo "WARNING: step 3b wasn't run this time. Below is the previous result, written $(mtime_of "$DIFF3")."
      cat "$DIFF3"
    else
      echo "(not run)"
    fi
    echo '```'
    echo
    echo "## 4. Stabilization quality metrics (cpstab.metrics, stride=$METRIC_STRIDE)"
    echo
    [ -e "$MET5" ] && cat "$MET5" || echo "(not run)"
    echo
    echo "How to read this: replicate_f64 shares its lineage with the serial/MATLAB port and is the baseline;"
    echo "improved_f64's four changes should lower residual motion and raise sharpness,"
    echo "while field noise ratio is a control quantity for sharpness, not a score (see cpstab/metrics.py)."
    echo "truth is the output of the original MATLAB pipeline, included only for side-by-side reference — its"
    echo "difference from this port has already been quantified separately in reportB/cpstab_validation.md."
  } > "$REPORT"
  say ""
  say "wrote $REPORT"
fi

say ""
say "bench_full.sh done $(date '+%F %T')"
say "  log    : $LOG"
say "  report : $REPORT"

# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Idempotent rather than "delete when done". Each step uses its own
#    artifact as the marker and skips if it exists. The shortest step in
#    this pipeline still takes tens of minutes; dying mid-run from
#    disk/OOM/an accidental Ctrl-C is the norm, and rerunning the same
#    command needs to pick up where it left off, not redo step 1's 60 GiB
#    conversion from scratch.
#    But "exists" alone isn't enough: nothing in an artifact's path can say
#    "the code changed". This actually happened — step 4's tif was 73
#    minutes older than improved.py, and rerunning the exact same command
#    produced output that differed in 95.7% of pixels, yet the script
#    SKIPped anyway and republished the stale score under today's date. So
#    the criterion is artifact + $BENCH/_stamp_<n>.sha (= sha256 of the code
#    files this step depends on + this invocation's command line); it only
#    skips when both match, and the fingerprint is only written after this
#    step hits rc=0.
#    Artifacts left over from before this refactor have no fingerprint and
#    fall back to mtime comparison (trusted only if the artifact is newer
#    than all dependent code), so step 1's 60 GiB doesn't get judged stale
#    right out of the gate; once trusted, the fingerprint gets backfilled.
#    Step 1's fingerprint is ADVISORY — rerunning it means overwriting
#    60 GiB, and relayout itself also refuses to overwrite, so
#    auto-rerunning would just turn into a false failure; a fingerprint
#    mismatch is loudly flagged instead, and a human decides via --force.
#    --force is still the global override switch; combine with --steps to
#    force just one step.
# 2. A failed step aborts immediately (rc!=0 -> exit). Steps 2, 3, 4 all
#    depend on step 1's storage, step 3's diff stats depend on step 2's
#    output, step 5 depends on 2 and 4; letting a failed step continue would
#    only produce a report that looks complete but is actually missing
#    pieces — exactly the kind of silent failure relayout's DESIGN NOTES 8
#    guards against, and the same applies at this orchestration layer.
#    Step 5 is the only step that doesn't go through run_step; its rc used
#    to be hardcoded to 0, so when metrics raised an exception the
#    traceback would get pasted into section 4 as if it were the metrics
#    table, while the timing table would still show rc=0. It now takes
#    PIPESTATUS[0] (the pipe's last stage is grep, and $? belongs to grep)
#    and aborts the same way.
# 2b. The report body only publishes what **actually ran this time**. §1's
#    timing table got a "source" column and only totals the steps that ran;
#    §2/§3 print a "previous run + written-at" warning instead of a raw cat
#    when the corresponding step didn't run. The old `[ -e "$CMP2" ] && cat`
#    "(not run)" fallback branch only triggered when the file had literally
#    never been generated — i.e. it only worked the very first time in
#    history. After that, a single `--steps 5` could produce a report
#    stamped with today's date and the current param header, while carrying
#    the previous round's bitwise-comparison conclusion and steps 1-4
#    timings. The author had already recognized this staleness category at
#    L106-108 and guarded the log grep with `ran 2`, but hadn't carried the
#    same discipline through to the report body.
# 3. Step 3's diff stats are streamed. Each output is 1.5 GB uint16; loading
#    both whole and computing correlation would need ~6 GB of
#    double-precision intermediates. Accumulating Pearson's six sums
#    (sa/sb/saa/sbb/sab/n) frame by frame gives an r numerically equivalent
#    to computing over the whole array, but with constant memory. max|diff|
#    is reported together with "how many pixels differ", because on the
#    validation subset float32's written TIFF turned out bitwise identical
#    — reporting only r=0.999... would bury the much stronger conclusion
#    that not a single pixel actually differed.
# 4. metrics defaults to stride 8. metrics.py's residual-motion term is a
#    pair of 2-D FFTs per frame; running all 1500 frames x 3 inputs would be
#    pure waste — its own docstring says 4-8 is enough. The key point is
#    that all three inputs use the same stride — the table's rows must be
#    comparable, so stride is a script-level variable rather than something
#    typed by hand each time.
# 5. Why truth is also in step 5's table. truth is the original MATLAB
#    pipeline's output, and it's not bitwise-related to step 2 (reportB
#    already quantified a systematic r~0.67 difference). It's in the same
#    table not to decide who's "right", but to give improved's
#    "steadier/sharper" claim a third-party yardstick: if improved beats
#    replicate on the metrics but also beats truth, that's more likely the
#    metric getting fooled by some kind of smoothing than the four changes
#    genuinely winning. In subset mode truth's shape doesn't match (1500 vs
#    40 frames), so it's skipped outright.
# 6. Why steps 2, 3, 4 don't run in parallel. Three fast_run instances each
#    spinning up 10 workers hitting the same disk would fight over the IO
#    queue, and the measured wall clock wouldn't be the true cost of any one
#    configuration — this pipeline's whole output is "timings", and serial
#    execution is a precondition for those timings being correct.
