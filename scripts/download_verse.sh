#!/usr/bin/env bash
# Fetch the VerSe 2020 dataset (CC BY-SA 4.0) from the official mirrors.
#
# Data is NOT vendored in this repository — this script is how a fresh clone
# obtains it. Sources are the links published by the dataset authors at
# https://github.com/anjany/verse (OSF project: https://osf.io/t98fz/).
#
# Usage:  scripts/download_verse.sh [target_dir]     # default: data/verse2020
#
# If you already hold a local copy, skip this and point the scripts at it:
#   python scripts/build_partition.py --data-root /path/to/verse2020
#
# License note: VerSe is CC BY-SA 4.0 (Sekuboyina et al., "VerSe: A Vertebrae
# labelling and segmentation benchmark", Medical Image Analysis 2021). This
# repo's CODE is MIT; the DATA keeps its own license and never enters git.

set -euo pipefail

TARGET="${1:-data/verse2020}"
BASE="https://s3.bonescreen.de/public/VerSe-complete"
ARCHIVES=(dataset-verse20training.zip dataset-verse20validation.zip dataset-verse20test.zip)

mkdir -p "$TARGET"
cd "$TARGET"

for a in "${ARCHIVES[@]}"; do
  if [ -f ".done-$a" ]; then
    echo "already extracted: $a"
    continue
  fi
  echo "downloading $a ..."
  curl -fL --retry 3 -o "$a" "$BASE/$a"
  sha256sum "$a" | tee -a checksums.sha256
  unzip -q -o "$a"
  rm -f "$a"
  touch ".done-$a"
done

echo "VerSe 2020 ready under: $TARGET"
echo "Recorded archive checksums: $TARGET/checksums.sha256"
