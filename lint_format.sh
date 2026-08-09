#!/usr/bin/env bash
set -euo pipefail

isort common model train utils lmms_eval -l 120
black common model train utils lmms_eval -l 120
