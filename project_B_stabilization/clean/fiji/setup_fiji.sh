#!/usr/bin/env bash
# setup_fiji.sh — fetch a Fiji.app into this folder for the self-contained pipeline.
#
# Downloads Fiji for the host OS, unpacks it to clean/fiji/Fiji.app, then adds the
# MultiStackReg plugin. Two pieces this script CANNOT fetch automatically (see NOTES.md):
#   - mij.jar           (the MATLAB<->ImageJ bridge; not shipped with Fiji)
#   - TurboRegHL_.jar   (Fred's custom TurboReg fork; needed only for optotune runs)
# Drop those into Fiji.app/jars/ and Fiji.app/plugins/ respectively.
#
# Usage:  bash setup_fiji.sh
set -euo pipefail
cd "$(dirname "$0")"

DEST="Fiji.app"
if [ -d "$DEST" ]; then
  echo "Fiji.app already present here. Delete it first to re-fetch. Skipping."
  exit 0
fi

case "$(uname -s)" in
  Darwin) URL="https://downloads.imagej.net/fiji/latest/fiji-macosx.zip" ;;
  Linux)  URL="https://downloads.imagej.net/fiji/latest/fiji-linux64.zip" ;;
  MINGW*|MSYS*|CYGWIN*) URL="https://downloads.imagej.net/fiji/latest/fiji-win64.zip" ;;
  *) echo "Unknown OS $(uname -s); download Fiji manually from https://fiji.sc and unzip here as Fiji.app"; exit 1 ;;
esac

echo "Downloading Fiji: $URL"
curl -L -o fiji.zip "$URL"
echo "Unpacking..."
unzip -q fiji.zip
rm -f fiji.zip
# the zip unpacks to 'Fiji.app' (mac/linux) — normalize if needed
[ -d "$DEST" ] || { d=$(find . -maxdepth 1 -type d -name 'Fiji*' | head -1); [ -n "$d" ] && mv "$d" "$DEST"; }

echo "Installing MultiStackReg plugin..."
curl -L -o "$DEST/plugins/MultiStackReg_.jar" \
  "https://github.com/miura/MultiStackRegistration/releases/download/1.45/MultiStackReg_.jar" || \
  echo "  (MultiStackReg download failed — install it from Fiji's Update site or https://github.com/miura/MultiStackRegistration)"

echo
echo "Done: $(pwd)/$DEST"
echo "STILL REQUIRED for MATLAB MIJ (see NOTES.md):"
echo "  - $DEST/jars/mij.jar       (MATLAB<->ImageJ bridge)"
echo "  - $DEST/plugins/TurboRegHL_.jar   (only for optotune/'affine' runs)"
echo
echo "Pin the version that produced ground truth before trusting registration output."
