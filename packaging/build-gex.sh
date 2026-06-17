#!/bin/bash
# Build gimp-segany.gex — a GIMP 3 extension package that installs the plug-in
# files with a double-click (Edit → Preferences → … opens such files as
# extensions in GIMP 3).
#
# IMPORTANT: a .gex can only ship the plug-in files. It does NOT provision the
# external Python backend (PyTorch + sam2) the plug-in needs — users still run
# install.command / install-linux.sh / install-windows.ps1 for that. The .gex
# is a convenience for the file-copy step only.
#
# Archive layout required by GIMP (validated on install):
#   <ext-id>/<ext-id>.metainfo.xml          AppStream id must equal <ext-id>
#   <ext-id>/seganyplugin/seganyplugin.py   folder name must match the .py
#   <ext-id>/seganyplugin/seganybridge.py
# All files live under the single top-level <ext-id> directory.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

EXT_ID="io.github.leiverkus.gimpsegany"
OUT="$REPO/gimp-segany.gex"

# Read the live version so the packaged metainfo can't drift from the code.
VERSION=$(grep -oE '__version__ = "[^"]+"' seganybridge.py | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
[ -n "$VERSION" ] || { echo "Error: could not read __version__ from seganybridge.py" >&2; exit 1; }

BUILD="$(mktemp -d)"
DEST="$BUILD/$EXT_ID"
mkdir -p "$DEST/seganyplugin"

cp seganyplugin.py seganybridge.py "$DEST/seganyplugin/"
chmod +x "$DEST/seganyplugin/seganyplugin.py"

# Metainfo at {id}/{id}.metainfo.xml; inject the live version into the release.
sed -E "s/(<release version=\")[^\"]+/\1${VERSION}/" \
    "packaging/${EXT_ID}.metainfo.xml" > "$DEST/${EXT_ID}.metainfo.xml"

rm -f "$OUT"
( cd "$BUILD" && zip -r -q "$OUT" "$EXT_ID" )
rm -rf "$BUILD"
echo "built $OUT (version $VERSION)"
