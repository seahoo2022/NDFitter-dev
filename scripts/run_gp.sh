#!/bin/sh
# Pass the data location explicitly; activate the desired environment first.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${PYTHON:-python3}" -m NDFitter.GPyTorch.train_yscaled "$@"
