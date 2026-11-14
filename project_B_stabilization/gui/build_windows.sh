#!/bin/bash
# Build the portable Windows bundle (no install needed on the target PC).
# Needs: uv, curl. Output: dist/ShipleyStabilizer.zip
set -euo pipefail
cd "$(dirname "$0")"
REL=20261026; PYV=3.12.16
F="cpython-$PYV+$REL-x86_64-pc-windows-msvc-install_only.tar.gz"
B=dist/ShipleyStabilizer
rm -rf dist; mkdir -p $B/engine
curl -sSL -o dist/$F "https://github.com/astral-sh/python-build-standalone/releases/download/$REL/${F/+/%2B}"
curl -sSL -o dist/SHA256SUMS "https://github.com/astral-sh/python-build-standalone/releases/download/$REL/SHA256SUMS"
(cd dist && grep " $F\$" SHA256SUMS | shasum -a 256 -c -)
tar xzf dist/$F -C $B
uv pip install -q --python-platform x86_64-pc-windows-msvc --python-version 3.12 \
  --target $B/python/Lib/site-packages numpy==2.5.4 scipy==1.18.1 tifffile==2026.10.15
rm -rf $B/python/Lib/site-packages/bin $B/python/Lib/site-packages/*.whl \
       $B/python/Lib/test $B/python/include $B/python/libs
find $B/python -name '*.pdb' -delete
cp app.py shipley_run.py oirread.py ../pyport/fast_run.py $B/engine/
rsync -a --exclude __pycache__ --exclude tests --exclude validate.py ../pyport/cpstab $B/engine/
cp *.bat WINDOWS_README.txt $B/
mv $B/WINDOWS_README.txt $B/README.txt
(cd dist && zip -qr -X ShipleyStabilizer.zip ShipleyStabilizer)
ls -la dist/ShipleyStabilizer.zip
