#!/bin/zsh
# OIR -> OME-TIFF (BigTIFF) via Bio-Formats bfconvert.
# Usage: ./convert_oir.sh <master.oir> <out.ome.tif> [extra bfconvert args...]
# Replaces the original pipeline's "FluoView manual export .tif.frames" step;
# multi-file chunk series (_00001...) are grouped automatically.
set -euo pipefail
export JAVA_HOME="$HOME/tools/jdk-17.0.20.1+1-jre/Contents/Home"
export BF_MAX_MEM=4g
BF="$HOME/tools/bftools"

in="$1"; out="$2"; shift 2
echo "== showinf metadata =="
"$BF/showinf" -nopix -novalid "$in" | grep -E "Width|Height|SizeZ|SizeT|SizeC|Dimension order|Pixel type|Series count" || true
echo "== convert =="
time "$BF/bfconvert" -bigtiff -nogroup "$@" "$in" "$out"
# note: OIR series auto-group by default; -nogroup prevents pulling in other
# runs from the same directory -- if only 1 T comes out, drop -nogroup and rerun
