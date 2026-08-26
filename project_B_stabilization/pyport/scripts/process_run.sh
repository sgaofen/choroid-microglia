#!/bin/bash
# ===========================================================================
# process_run.sh — one-click batch driver for a single experiment run
#   (oir volumes -> raw.ome.tif -> T-major .npy -> fast_run -> metrics report)
# ===========================================================================
# For batch-processing the remaining experiment runs. Five fixed steps:
#
#   1 bfconvert   oir volume dir -> <out>/raw.ome.tif            (Bio-Formats)
#   2 relayout    raw.ome.tif  -> <out>/raw.tzcyx.npy --verify 16
#   3 cleanup     delete raw.ome.tif once npy verification passes (the only
#                 deletion allowed anywhere in this script)
#   4 fast_run    raw.tzcyx.npy -> <out>/replicate/*_mean_zproj.tif
#   5 metrics     <out>/replicate/*_mean_zproj.tif -> <out>/report.md
#
# Usage
# -----
#   scripts/process_run.sh <oir_volume_dir> <out_dir> [--dry-run] [--workers N]
#
#   --dry-run         only print the command each step would run — don't run
#                     anything, create dirs, or delete files
#   --workers N       process count for fast_run (default 10)
#   --skip-convert F  rehearsal/regression only: skip step 1's bfconvert and
#                     stand file F in for <out>/raw.ome.tif — landed via a
#                     symlink (out/raw.ome.tif -> F), not a copy, so step 3's
#                     "delete raw.ome.tif" only removes the link itself; F's
#                     rehearsal data is untouched. Never use this flag for
#                     production batches — it exists purely to run steps
#                     2-3-4-5 end to end without touching any full-scale
#                     data (see the "rehearsal mode" note below).
#
# Resumability
# ------------
# A step counts as "done" only when **both** the artifact file exists **and**
# <artifact>.ok exists; checking existence alone gets fooled by half-written
# files left behind by a power loss/Ctrl-C (a lesson learned the hard way in
# bench_full.sh — see the comment at the top of that file). .ok is only
# written after the corresponding command exits rc=0, so a half-written
# artifact can never pass this double check.
#   * artifact missing                    -> run the step normally
#   * artifact exists AND .ok exists      -> skip
#   * artifact exists but .ok missing     -> state untrusted (could be a
#     half-written artifact, or a complete one whose .ok never got written
#     in time). The script won't delete anything except step 3's ome.tif,
#     and won't decide "overwrite" vs. "trust" on your behalf — it errors
#     out and waits for manual review.
#
# Disk precheck: before starting, `df` free space must be >= 2.2x the oir
# dir size (or the --skip-convert stand-in file's size in rehearsal mode);
# if not, it errors out immediately without entering any step.
#
# All logs (the script's own progress lines + child stdout/stderr) are teed
# to <out>/process.log.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYPORT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="${PY:-python}"
export JAVA_HOME="${JAVA_HOME:-$HOME/tools/jdk-17.0.20.1+1-jre/Contents/Home}"
export BF_MAX_MEM="${BF_MAX_MEM:-4g}"
BFCONVERT="$HOME/tools/bftools/bfconvert"
export PYTHONPATH="$PYPORT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  sed -n '1,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- argument parsing -------------------------------------------------------
OIR_DIR=""
OUT_DIR=""
DRY_RUN=0
WORKERS=10
SKIP_CONVERT=""

POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --skip-convert) SKIP_CONVERT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [ "${#POSITIONAL[@]}" -lt 2 ]; then
  echo "usage: $(basename "${BASH_SOURCE[0]}") <oir_volume_dir> <out_dir> [--dry-run] [--workers N] [--skip-convert F]" >&2
  exit 2
fi
OIR_DIR="${POSITIONAL[0]}"
OUT_DIR="${POSITIONAL[1]}"

case "$WORKERS" in
  ''|*[!0-9]*) echo "--workers must be a positive integer, got '$WORKERS'" >&2; exit 2 ;;
esac

# ---- derived paths ----------------------------------------------------------
LOG="$OUT_DIR/process.log"
REPORT="$OUT_DIR/report.md"

RAW_OME="$OUT_DIR/raw.ome.tif"
RAW_OME_OK="$RAW_OME.ok"

RAW_NPY="$OUT_DIR/raw.tzcyx.npy"
RAW_NPY_OK="$RAW_NPY.ok"

STEP3_OK="$OUT_DIR/raw.ome.tif.deleted.ok"

REPLICATE_DIR="$OUT_DIR/replicate"
# out_base() = <out_dir>/<basename(raw npy) minus one extension> (config.py
# RegistrationConfig.out_base) -> "raw.tzcyx"; the zproj filename is fixed as
# "<out_base>_<proj_type>_zproj.tif", proj_type defaults to 'mean'.
ZPROJ="$REPLICATE_DIR/raw.tzcyx_mean_zproj.tif"
ZPROJ_OK="$ZPROJ.ok"

REPORT_OK="$REPORT.ok"

# ---- helpers ------------------------------------------------------------
# By the point this is called, LOG's directory should already exist (mkdir
# happens below in the "opening" section), so say() can tee -a unconditionally.
say() { echo "$@" | tee -a "$LOG"; }

now() { date '+%F %T'; }

# artifact + artifact.ok both present -> "skip"; both absent -> "run"; only
# artifact present (no .ok) -> "halt" (state untrusted). Pure predicate — does
# no say/exit itself, since callers almost always do `gate=$(step_gate ...)`,
# a subshell; an earlier version `exit 1`'d directly in the "halt" branch,
# which only ended the subshell — the outer script's $gate got an empty
# string back, fell into the `else` branch, and ran it again as "run" — the
# hard-stop branch never actually did anything (DESIGN NOTES 8). The real
# say + exit lives in halt_untrusted(), which the caller invokes explicitly
# from the main shell.
step_gate() {
  local artifact="$1" ok="$2"
  if [ -e "$artifact" ] && [ -e "$ok" ]; then
    echo "skip"
  elif [ -e "$artifact" ] && [ ! -e "$ok" ]; then
    echo "halt"
  else
    echo "run"
  fi
}

# Must be called from the main shell (not wrapped in $(...)), so exit 1 here
# actually terminates the script.
halt_untrusted() {
  local label="$1" artifact="$2" ok="$3"
  say "[$label] artifact exists but no .ok marker — state untrusted (could be a"
  say "  half-written artifact, or a complete one that just didn't get its marker written in time)."
  say "  artifact: $artifact"
  say "  This script won't auto-overwrite or delete it (nothing gets deleted except step 3's raw.ome.tif)."
  say "  After manual review: confirmed half-written -> delete the file by hand and rerun;"
  say "             confirmed intact  -> manually touch '$ok' and rerun."
  exit 1
}

# run_piped <step number> <name> <cmdline string for the log> -- <actual argv...>
# Runs a command with stdout/stderr both going to process.log; on failure
# (rc!=0) prints diagnostics and exits immediately (implements the hard
# "fail fast" requirement; uses PIPESTATUS[0] rather than tee's own rc — see
# the same trick in bench_full.sh).
run_piped() {
  local n="$1" name="$2"; shift 2
  say ""
  say "==========================================================="
  say "[step $n] $name"
  say "  started: $(now)"
  say "  cmd: $*"
  say "==========================================================="
  local t0 t1 rc
  t0=$(date +%s)
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  t1=$(date +%s)
  say "  [step $n] rc=$rc  wall=$((t1 - t0))s"
  STEP_SECONDS[$n]=$((t1 - t0))
  STEP_NAME[$n]="$name"
  if [ "$rc" -ne 0 ]; then
    say "  [step $n] FAILED — later steps depend on it, aborting (set -euo pipefail)."
    exit "$rc"
  fi
}

# plain (non-associative) arrays indexed 1..5 — macOS ships bash 3.2, which
# has no `declare -A`; regular arrays accept arbitrary integer subscripts.
STEP_SECONDS=()
STEP_NAME=()

# ---- argument validation -----------------------------------------------------------
if [ -n "$SKIP_CONVERT" ] && [ ! -e "$SKIP_CONVERT" ]; then
  echo "--skip-convert file does not exist: $SKIP_CONVERT" >&2
  exit 2
fi
if [ -z "$SKIP_CONVERT" ] && [ ! -d "$OIR_DIR" ]; then
  echo "oir volume dir does not exist: $OIR_DIR" >&2
  exit 2
fi

# ---- disk precheck: free space >= 2.2 x (oir dir size | rehearsal stand-in file size) ----------
# df targets <out>/, but with --dry-run the dir may not exist yet (dry-run
# doesn't create dirs) — walk up the path to the first existing ancestor dir
# and df that instead; almost always the same filesystem.
df_target="$OUT_DIR"
while [ ! -d "$df_target" ]; do
  df_target="$(dirname "$df_target")"
done

if [ -n "$SKIP_CONVERT" ]; then
  SRC_KB=$(du -sk "$SKIP_CONVERT" | awk '{print $1}')
  SRC_DESC="--skip-convert stand-in file $SKIP_CONVERT (rehearsal mode, stands in for oir dir size)"
else
  SRC_KB=$(du -sk "$OIR_DIR" | awk '{print $1}')
  SRC_DESC="oir dir $OIR_DIR"
fi
AVAIL_KB=$(df -k "$df_target" | awk 'NR==2 {print $4}')
NEED_KB=$(awk -v s="$SRC_KB" 'BEGIN{printf "%.0f", s*2.2}')

echo "process_run.sh  $(now)"
echo "  oir volume dir : $OIR_DIR"
echo "  out dir        : $OUT_DIR"
echo "  workers        : $WORKERS"
echo "  dry-run        : $DRY_RUN"
[ -n "$SKIP_CONVERT" ] && echo "  *** rehearsal mode: --skip-convert=$SKIP_CONVERT (skipping step 1's real bfconvert) ***"
echo "  disk precheck  : $SRC_DESC = ${SRC_KB} KB, need >= 2.2x = ${NEED_KB} KB,"
echo "                   available($df_target) = ${AVAIL_KB} KB"
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
  echo "  disk precheck failed: available space is under 2.2x, exiting (no step entered)." >&2
  exit 1
fi
echo "  disk precheck passed"

if [ "$DRY_RUN" -eq 1 ]; then
  echo ""
  echo "*** --dry-run: the following only prints each step's command, doesn't"
  echo "*** run bfconvert/relayout/fast_run/metrics, and deletes no files. It"
  echo "*** still creates the output dir and writes process.log like a normal"
  echo "*** run (same convention as scripts/bench_full.sh's --dry-run: creating"
  echo "*** dirs/writing logs doesn't count as 'running a step', see DESIGN NOTES 7)."
fi

# ---- opening ----------------------------------------------------------
# mkdir/writing process.log don't count as "running a step" (DESIGN NOTES 7),
# so they happen under --dry-run too; otherwise say() below would just fail
# with a tee -a error while the dir doesn't exist yet, and we wouldn't even
# get the command preview.
mkdir -p "$OUT_DIR" "$REPLICATE_DIR"
say ""
say "###########################################################"
say "# process_run.sh  $(now)"
say "#   oir volume dir : $OIR_DIR"
say "#   out dir        : $OUT_DIR"
say "#   workers=$WORKERS  python=$PY  dry_run=$DRY_RUN"
[ -n "$SKIP_CONVERT" ] && say "#   *** rehearsal mode: --skip-convert=$SKIP_CONVERT ***"
say "#   host        : $(hostname)"
say "###########################################################"

# ===========================================================================
# 1 bfconvert: oir volumes -> raw.ome.tif
# ===========================================================================
# Step 1's artifact gets legitimately deleted by step 3 afterward, so it
# can't just use the generic step_gate: "artifact missing" doesn't
# necessarily mean "hasn't run yet" for step 1 — it could also mean "ran,
# and step 3 already cleaned it up". Use $STEP3_OK to tell that third state
# apart from "never ran".
if [ -e "$RAW_OME_OK" ] && { [ -e "$RAW_OME" ] || [ -e "$STEP3_OK" ]; }; then
  gate1="skip"
elif [ -e "$RAW_OME_OK" ]; then
  # .ok present, but neither raw.ome.tif nor step 3's cleanup marker is —
  # not the "step 3 already cleaned it up normally" state; raw.ome.tif went
  # missing for some other reason.
  say "[step 1] inconsistent state: $RAW_OME_OK exists, but neither $RAW_OME"
  say "  nor the cleanup marker $STEP3_OK exist. Normally raw.ome.tif only"
  say "  disappears once step 3 succeeds, and that always leaves $STEP3_OK"
  say "  behind. Review manually (did someone delete raw.ome.tif by hand?)"
  say "  then decide: touch \"$STEP3_OK\" (if step 3 did it), or delete"
  say "  \"$RAW_OME_OK\" to let step 1 rerun (if it didn't)."
  exit 1
elif [ -e "$RAW_OME" ]; then
  gate1="halt"
else
  gate1="run"
fi
if [ "$gate1" = "skip" ]; then
  say "[step 1] bfconvert — SKIP (step 1 already done: $RAW_OME_OK present, and $RAW_OME either still there or cleaned up normally by step 3)"
elif [ "$gate1" = "halt" ]; then
  halt_untrusted "step 1" "$RAW_OME" "$RAW_OME_OK"
elif [ -n "$SKIP_CONVERT" ]; then
  # Rehearsal mode: skip bfconvert, stand in raw.ome.tif with a symlink. The
  # link points at the rehearsal data itself; when step 3 deletes it, it's
  # deleting the link, not the target — the 40-frame subset is untouched.
  say ""
  say "==========================================================="
  say "[step 1] bfconvert — SKIPPED (rehearsal mode --skip-convert)"
  say "  ln -s '$SKIP_CONVERT' '$RAW_OME'"
  say "==========================================================="
  if [ "$DRY_RUN" -eq 1 ]; then
    say "  (dry-run, not executed)"
  else
    t0=$(date +%s)
    ln -s "$SKIP_CONVERT" "$RAW_OME"
    touch "$RAW_OME_OK"
    t1=$(date +%s)
    STEP_SECONDS[1]=$((t1 - t0))
    STEP_NAME[1]="bfconvert (SKIPPED — rehearsal symlink to $SKIP_CONVERT)"
    say "  [step 1] symlink created, marked $RAW_OME_OK  wall=$((t1 - t0))s"
  fi
else
  FIRST_OIR=$(find "$OIR_DIR" -maxdepth 1 -type f -name '*.oir' | sort | head -n1)
  if [ -z "$FIRST_OIR" ]; then
    echo "no .oir file found under $OIR_DIR" >&2
    exit 1
  fi
  say "[step 1] bfconvert input, first volume file: $FIRST_OIR"
  say "  (Bio-Formats auto-chains the rest of the volumes by their naming in"
  say "   the same dir; if the conversion comes out with only 1 T, drop/check"
  say "   -nogroup and rerun — a known gotcha in scripts/convert_oir.sh)"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "  cmd: $BFCONVERT -bigtiff -nogroup '$FIRST_OIR' '$RAW_OME'"
    say "  (dry-run, not executed)"
  else
    run_piped 1 "bfconvert oir -> raw.ome.tif" \
      "$BFCONVERT" -bigtiff -nogroup "$FIRST_OIR" "$RAW_OME"
    touch "$RAW_OME_OK"
    say "  [step 1] marked $RAW_OME_OK"
  fi
fi

# ===========================================================================
# 2 relayout: raw.ome.tif -> raw.tzcyx.npy (T-major), --verify 16
# ===========================================================================
gate=$(step_gate "$RAW_NPY" "$RAW_NPY_OK")
if [ "$gate" = "skip" ]; then
  say "[step 2] relayout — SKIP (artifact+ok both present: $RAW_NPY)"
elif [ "$gate" = "halt" ]; then
  halt_untrusted "step 2" "$RAW_NPY" "$RAW_NPY_OK"
else
  if [ "$DRY_RUN" -eq 1 ]; then
    say ""
    say "[step 2] relayout — (dry-run, not executed)"
    say "  cmd: $PY '$PYPORT/scripts/relayout.py' --in '$RAW_OME' --out '$RAW_NPY' --verify 16"
  else
    run_piped 2 "relayout raw.ome.tif -> raw.tzcyx.npy (--verify 16)" \
      "$PY" "$PYPORT/scripts/relayout.py" \
        --in "$RAW_OME" --out "$RAW_NPY" --verify 16
    touch "$RAW_NPY_OK"
    say "  [step 2] marked $RAW_NPY_OK"
  fi
fi

# ===========================================================================
# 3 cleanup: once npy's .ok is confirmed written, delete raw.ome.tif
#    (the only deletion allowed anywhere in this script; must happen after
#    relayout --verify succeeds)
# ===========================================================================
if [ -e "$STEP3_OK" ]; then
  if [ -e "$RAW_OME" ]; then
    echo "[step 3] inconsistent state: $STEP3_OK exists (marked cleaned up) but $RAW_OME is still there." >&2
    echo "  This shouldn't happen (did someone put raw.ome.tif back?) — review manually, this script won't handle it automatically." >&2
    exit 1
  fi
  say "[step 3] cleanup raw.ome.tif — SKIP (already cleaned up: $STEP3_OK)"
elif [ "$DRY_RUN" -eq 1 ]; then
  say ""
  say "[step 3] cleanup — (dry-run, not executed)"
  say "  cmd: rm '$RAW_OME'   # only runs once $RAW_NPY_OK exists"
else
  if [ ! -e "$RAW_NPY_OK" ]; then
    # run_piped already exits on step 2's failure, so normal flow never
    # reaches here; kept as a defensive check.
    echo "[step 3] $RAW_NPY_OK does not exist, npy hasn't been verified yet, refusing to delete $RAW_OME" >&2
    exit 1
  fi
  say ""
  say "==========================================================="
  say "[step 3] cleanup: deleting $RAW_OME (npy already verified; the only deletion allowed in this script)"
  say "  started: $(now)"
  say "==========================================================="
  t0=$(date +%s)
  if [ -e "$RAW_OME" ] || [ -L "$RAW_OME" ]; then
    rm "$RAW_OME"
  fi
  touch "$STEP3_OK"
  t1=$(date +%s)
  STEP_SECONDS[3]=$((t1 - t0))
  STEP_NAME[3]="cleanup raw.ome.tif"
  say "  [step 3] deleted $RAW_OME, marked $STEP3_OK  wall=$((t1 - t0))s"
fi

# ===========================================================================
# 4 fast_run: raw.tzcyx.npy -> replicate/*_mean_zproj.tif
#    (float64 / replicate — faithful MATLAB port, bitwise-identical to the serial version)
# ===========================================================================
gate=$(step_gate "$ZPROJ" "$ZPROJ_OK")
if [ "$gate" = "skip" ]; then
  say "[step 4] fast_run — SKIP (artifact+ok both present: $ZPROJ)"
elif [ "$gate" = "halt" ]; then
  halt_untrusted "step 4" "$ZPROJ" "$ZPROJ_OK"
else
  if [ "$DRY_RUN" -eq 1 ]; then
    say ""
    say "[step 4] fast_run — (dry-run, not executed)"
    say "  cmd: $PY '$PYPORT/fast_run.py' --raw '$RAW_NPY' --out-dir '$REPLICATE_DIR'" \
        "--workers $WORKERS --dtype float64 --mode replicate"
  else
    run_piped 4 "fast_run npy float64 replicate (workers=$WORKERS)" \
      "$PY" "$PYPORT/fast_run.py" \
        --raw "$RAW_NPY" --out-dir "$REPLICATE_DIR" \
        --workers "$WORKERS" --dtype float64 --mode replicate
    if [ ! -e "$ZPROJ" ]; then
      echo "[step 4] fast_run rc=0 but expected artifact is missing: $ZPROJ" >&2
      echo "  (did the out_base naming contract change? see scripts/relayout.py DESIGN NOTES 9)" >&2
      exit 1
    fi
    touch "$ZPROJ_OK"
    say "  [step 4] marked $ZPROJ_OK"
  fi
fi

# ===========================================================================
# 5 metrics: replicate/*_mean_zproj.tif -> report.md (+ per-step timings)
# ===========================================================================
gate=$(step_gate "$REPORT" "$REPORT_OK")
if [ "$gate" = "skip" ]; then
  say "[step 5] cpstab.metrics — SKIP (artifact+ok both present: $REPORT)"
elif [ "$gate" = "halt" ]; then
  halt_untrusted "step 5" "$REPORT" "$REPORT_OK"
else
  METRIC_CMD=("$PY" -m cpstab.metrics --stride 8 "replicate=$ZPROJ")
  if [ "$DRY_RUN" -eq 1 ]; then
    say ""
    say "[step 5] cpstab.metrics — (dry-run, not executed)"
    say "  cmd: ${METRIC_CMD[*]} > '$REPORT'"
  else
    say ""
    say "==========================================================="
    say "[step 5] cpstab.metrics (stride=8)"
    say "  started: $(now)"
    say "  cmd: ${METRIC_CMD[*]} > $REPORT"
    say "==========================================================="
    t0=$(date +%s)
    set +e
    "${METRIC_CMD[@]}" > "$REPORT" 2> >(tee -a "$LOG" >&2)
    rc=$?
    set -e
    t1=$(date +%s)
    say "  [step 5] rc=$rc  wall=$((t1 - t0))s"
    if [ "$rc" -ne 0 ]; then
      say "  [step 5] FAILED — $REPORT is not a complete metrics table, no marker written, aborting."
      exit "$rc"
    fi
    STEP_SECONDS[5]=$((t1 - t0))
    STEP_NAME[5]="cpstab.metrics"
    cat "$REPORT" >> "$LOG"     # report content also goes into process.log (hard requirement: all logs get teed)

    # ---- append per-step timings (only the steps that actually ran this call; skipped steps have no fresh wall-clock number) --
    {
      echo ""
      echo "## Step timings (this invocation)"
      echo ""
      echo "| # | step | wall (s) |"
      echo "|---|---|---|"
      for n in 1 2 3 4 5; do
        if [ -n "${STEP_NAME[$n]:-}" ]; then
          printf '| %s | %s | %s |\n' "$n" "${STEP_NAME[$n]}" "${STEP_SECONDS[$n]}"
        else
          printf '| %s | (skipped this run — product+.ok already present) |  |\n' "$n"
        fi
      done
    } >> "$REPORT"
    cat >> "$LOG" <<EOF

## Step timings (this invocation)
$(for n in 1 2 3 4 5; do
    if [ -n "${STEP_NAME[$n]:-}" ]; then
      echo "  [$n] ${STEP_NAME[$n]}: ${STEP_SECONDS[$n]}s"
    else
      echo "  [$n] skipped this run"
    fi
  done)
EOF
    touch "$REPORT_OK"
    say "  [step 5] marked $REPORT_OK"
  fi
fi

say ""
say "process_run.sh done $(now)"
say "  log    : $LOG"
say "  report : $REPORT"

# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Double-checking artifact+.ok, not just artifact existence. A lesson
#    recorded in bench_full.sh's top comment: a power loss/Ctrl-C during
#    step 1's 60 GB conversion leaves behind a file that looks complete but
#    is actually half-written; checking "exists" alone would treat it as
#    done. .ok is a second gate here, only written after the corresponding
#    command hits rc=0 (for step 3, that means "the delete has actually
#    completed") — a half-written artifact can never get its .ok. The third
#    state, "artifact present but .ok missing", is never silently treated as
#    either "skip" or "overwrite" — it hard-stops for manual review instead:
#    overwriting would mean deleting/rewriting a file whose state is
#    unknown, and the hard requirement is "delete nothing except step 3's
#    ome.tif" — this script isn't authorized to make that call on your behalf.
# 2. Only step 3 deletes anything, and only after step 2's .ok is already in
#    hand (written once relayout --verify 16 passes). The delete itself has
#    its own .ok (raw.ome.tif.deleted.ok), so a rerun won't try rm again on a
#    file that's already gone; if the "already cleaned up" marker exists but
#    raw.ome.tif has reappeared, that's treated as an inconsistent state and
#    hard-stopped (no guessing intent).
# 3. --skip-convert is for rehearsal/regression use only. It stands in for
#    <out>/raw.ome.tif with a symlink rather than a copy: step 3 deletes the
#    link itself (rm on a symlink doesn't touch what it points to), so the
#    rehearsal data (e.g. the 40-frame subset) is never corrupted or deleted
#    by this pipeline — "never touch any full-scale or rehearsal data" is
#    guaranteed by filesystem semantics, not by script discipline.
# 4. The disk precheck runs as soon as the oir dir's size (or the
#    --skip-convert stand-in file's size) is known, and can exit before
#    entering any step — you'll never see "ran part of step 2's conversion
#    and only then discovered there's not enough disk". In rehearsal mode
#    the stand-in file's size substitutes for the oir dir's size, so
#    rehearsal actually exercises this gate too instead of bypassing it.
# 5. fast_run.py itself protects the final artifact (write_zproj_tiff) with
#    a "refuse to overwrite if it already exists" check (io_rw.py's force
#    check), but not intermediate state (.dftshifts.npz etc.) — that's
#    written with a plain open(...,'wb') truncate, so a rerun mid-step-4 is
#    safe. This script doesn't clean up anything under replicate/ outside of
#    step 4 (hard requirement: delete nothing except in step 3), so the rare
#    edge case where "step 4 finished writing its final artifact but the
#    script didn't get to touch .ok in time" falls into note 1's hard-stop
#    branch, and needs a manual touch of $ZPROJ.ok after confirmation.
# 6. Per-step timings only count steps that **actually ran this invocation**
#    (STEP_SECONDS only gets filled when run_piped / step 3 / step 5 actually
#    finish executing); skipped steps are honestly labeled "skipped this
#    run" in the timing table rather than reusing an old number to pass off
#    as this run's wall clock (the same "the report only publishes what
#    actually ran this time" discipline as bench_full.sh — see its
#    DESIGN NOTES 2b).
# 7. --dry-run still mkdirs the output dir and still writes process.log (a
#    lesson from rehearsal: the first version wrapped both mkdir and
#    say()/tee inside `if [ DRY_RUN -eq 0 ]`, so a not-yet-existing <out>
#    dir with --dry-run crashed on the very first say() with `tee: No such
#    file or directory`, without even printing the command preview).
#    "--dry-run only prints commands, doesn't execute" applies to the five
#    real actions (bfconvert/relayout/fast_run/metrics) that cost time or
#    change the world outside $OUT_DIR — not to creating an empty dir or
#    writing its own log inside its own output dir. bench_full.sh's
#    --dry-run follows the same convention (its `mkdir -p "$BENCH" ...` also
#    runs unconditionally before the --dry-run branch).
# 8. step_gate()'s "halt" branch must not exit itself. Hit this in
#    rehearsal: callers almost always do `gate=$(step_gate ...)`, and
#    `$(...)` is a subshell — `exit 1` inside it only ends the subshell; the
#    outer script's `$gate` only gets whatever stdout the subshell produced
#    before exiting (here, an empty string), so the caller's
#    `if [ "$gate" = "skip" ]` can't tell it was "halt" and falls into the
#    `else` branch, running normally — "halt if artifact exists without
#    .ok", meant to be the last safety net, did nothing at all under this
#    pattern. step_gate() is now a pure predicate (only echoes
#    "skip"/"run"/"halt", never says/exits); the real diagnostics and exit
#    moved into halt_untrusted(), which callers invoke explicitly from the
#    **main shell**. The same trap had a second instance: step 1's own
#    completion check can't just copy the generic "artifact+.ok both
#    present" rule, because step 1's artifact gets legitimately deleted by
#    step 3 afterward — "raw.ome.tif missing" could mean "hasn't run yet"
#    for step 1, or it could mean "ran, and step 3 already cleaned it up
#    normally" — the two must be told apart with $STEP3_OK, or a rerun will,
#    after step 3 has already deleted it once, try to recreate it against a
#    now-nonexistent raw.ome.tif (reproduced in rehearsal by running this
#    pipeline twice in a row: the second invocation treated step 1 as
#    "never ran" again and recreated the symlink, and step 3 immediately
#    reported the inconsistent state "cleanup marker present but
#    raw.ome.tif reappeared" and hard-stopped).
