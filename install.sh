#!/usr/bin/env bash
# Installs yaehmop-bandfit in its own conda environment.
#
#   ./install.sh [options]
#
#   --env NAME              conda environment name (default: yaehmop-bandfit)
#   --yaehmop-bind EXE      existing YAeHMOP bind executable        -> stored in the environment
#   --yaehmop-params FILE   existing YAeHMOP eht_parms.dat          -> stored in the environment
#   --build-yaehmop         download and compile YAeHMOP into the environment (bind + eht_parms.dat),
#                           from the fork github.com/ac-dias/yaehmop (double-zeta p, spin, SOC)
#   --yaehmop-repo URL      git repository for --build-yaehmop (default: the ac-dias/yaehmop fork)
#   --yaehmop-ref REF       git commit/tag for --build-yaehmop (default: d7e3deb, the tested one)
#   --tests                 run the test suite at the end
#   -h, --help              this help
#
# Without --yaehmop-* or --build-yaehmop, YAEHMOP_BIND / YAEHMOP_PARAM_FILE from the current shell
# are kept if they point to existing files. YAeHMOP is only needed for fitting and recomputing bands.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="yaehmop-bandfit"
BIND="" PARAMS=""
BUILD_YAEHMOP=0
# Fork of YAeHMOP with double-zeta p orbitals (needed by basis_dz = X:p), spin polarization and
# SOC (also SOC on two spin tables, with charge iteration); with single-zeta tables it gives the
# same bands as the original program. spin_joint_fit.py needs its --parm-up/--common-fermi.
YAEHMOP_REPO="https://github.com/ac-dias/yaehmop.git"
YAEHMOP_REF="d7e3debf8125a37336d1fd41f0d25eb32f9d04ec"
YAEHMOP_SRC="$HERE/third_party/yaehmop"
RUN_TESTS=0

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --yaehmop-bind) BIND="$2"; shift 2 ;;
    --yaehmop-params) PARAMS="$2"; shift 2 ;;
    --build-yaehmop) BUILD_YAEHMOP=1; shift ;;
    --yaehmop-ref) YAEHMOP_REF="$2"; shift 2 ;;
    --yaehmop-repo) YAEHMOP_REPO="$2"; shift 2 ;;
    --tests) RUN_TESTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

if [ "$BUILD_YAEHMOP" = 1 ] && { [ -n "$BIND" ] || [ -n "$PARAMS" ]; }; then
  die "use either --build-yaehmop or --yaehmop-bind/--yaehmop-params, not both"
fi
[ -z "$BIND" ] || BIND="$(realpath "$BIND")"
[ -z "$PARAMS" ] || PARAMS="$(realpath "$PARAMS")"
[ -z "$BIND" ] || [ -x "$BIND" ] || die "not an executable: $BIND"
[ -z "$PARAMS" ] || [ -f "$PARAMS" ] || die "file not found: $PARAMS"

# ---------------------------------------------------------------- conda
if ! command -v conda >/dev/null 2>&1; then
  for c in /opt/anaconda3 "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || die "conda not found; install Miniforge (https://github.com/conda-forge/miniforge)"
source "$(conda info --base)/etc/profile.d/conda.sh"
SOLVER=conda; command -v mamba >/dev/null 2>&1 && SOLVER=mamba
# the classic conda solver can take very long with conda-forge: use libmamba when available
if [ "$SOLVER" = conda ] && "$(conda info --base)/bin/python" -c "import conda_libmamba_solver" 2>/dev/null; then
  export CONDA_SOLVER=libmamba
fi
export CONDA_CHANNEL_PRIORITY=strict

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "updating conda environment $ENV_NAME"
  $SOLVER env update -q -n "$ENV_NAME" -f "$HERE/environment.yml"
else
  say "creating conda environment $ENV_NAME"
  $SOLVER env create -q -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
set +u; conda activate "$ENV_NAME"; set -u

# ---------------------------------------------------------------- Python requirements
say "checking the Python requirements"
python -m pip install -q -r "$HERE/requirements.txt"
python -c "import numpy, scipy, ase, PIL, matplotlib, yaml, sklearn, cma" \
  || die "Python packages missing in $ENV_NAME"
python -c "import PySide6" 2>/dev/null \
  || say "warning: PySide6 not importable; the desktop interface band_fit_gui.py will not start"

# ---------------------------------------------------------------- YAeHMOP
if [ "$BUILD_YAEHMOP" = 1 ]; then
  say "building YAeHMOP $YAEHMOP_REF into $CONDA_PREFIX"
  $SOLVER install -q -y -n "$ENV_NAME" -c conda-forge c-compiler cmake make liblapack libblas
  set +u; conda activate "$ENV_NAME"; set -u      # reload compiler activation scripts
  if [ ! -d "$YAEHMOP_SRC/.git" ]; then
    mkdir -p "$(dirname "$YAEHMOP_SRC")"
    git clone -q "$YAEHMOP_REPO" "$YAEHMOP_SRC"
  fi
  git -C "$YAEHMOP_SRC" remote set-url origin "$YAEHMOP_REPO"
  git -C "$YAEHMOP_SRC" fetch -q origin || true
  git -C "$YAEHMOP_SRC" checkout -q "$YAEHMOP_REF"
  PARM_DIR="$CONDA_PREFIX/share/yaehmop"
  mkdir -p "$PARM_DIR"
  cp "$YAEHMOP_SRC/tightbind/eht_parms.dat" "$PARM_DIR/eht_parms.dat"
  BUILD="$YAEHMOP_SRC/_build_$ENV_NAME"
  # GCC >= 14 turns these old-C warnings of the YAeHMOP sources into errors
  CFLAGS_OLD_C="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion"
  cmake -S "$YAEHMOP_SRC/tightbind" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="$CFLAGS_OLD_C" \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
        -DUSE_BLAS_LAPACK=ON -DPARM_FILE_LOC="$PARM_DIR/eht_parms.dat" >/dev/null
  cmake --build "$BUILD" -- -j"$(nproc)" >/dev/null
  cmake --install "$BUILD" >/dev/null
  BIND="$CONDA_PREFIX/bin/bind"
  PARAMS="$PARM_DIR/eht_parms.dat"
  [ -x "$BIND" ] || die "YAeHMOP build did not produce $BIND"
fi

# keep the variables of the current shell when nothing else was given
if [ -z "$BIND" ] && [ -n "${YAEHMOP_BIND:-}" ] && [ -x "${YAEHMOP_BIND}" ]; then BIND="$YAEHMOP_BIND"; fi
if [ -z "$PARAMS" ] && [ -n "${YAEHMOP_PARAM_FILE:-}" ] && [ -f "${YAEHMOP_PARAM_FILE}" ]; then PARAMS="$YAEHMOP_PARAM_FILE"; fi

if [ -n "$BIND" ] && [ -n "$PARAMS" ]; then
  say "storing YAEHMOP_BIND and YAEHMOP_PARAM_FILE in $ENV_NAME"
  conda env config vars set -n "$ENV_NAME" YAEHMOP_BIND="$BIND" YAEHMOP_PARAM_FILE="$PARAMS" >/dev/null
  echo "  YAEHMOP_BIND=$BIND"
  echo "  YAEHMOP_PARAM_FILE=$PARAMS"
  # smoke test: one real calculation (H2 molecule)
  T="$(mktemp -d)"
  printf 'H2\n\nGeometry\n2\n1 H 0 0 0\n2 H 0 0 0.74\n\nmolecular\n\ncharge\n0\n' > "$T/h2.bind"
  if "$BIND" "$T/h2.bind" "$PARAMS" >/dev/null 2>&1 && grep -q "Total_Energy" "$T/h2.bind.out"; then
    echo "  bind works: $(grep -m1 Total_Energy "$T/h2.bind.out")"
  else
    echo "  WARNING: bind did not run correctly; see $T" >&2
  fi
  # double-zeta p (basis_dz = X:p) needs the fork: its .out lists such a shell as "2P DZ:"
  printf 'C2\n\nGeometry\n2\n1 C 0 0 0\n2 C 0 0 1.30\n\nmolecular\n\ncharge\n0\n' > "$T/c2dz.bind"
  printf 'C 6 4 1 2 s -21.4 1.625 0.0 1.0 0.0\nC 6 4 2 2 p -11.4 1.9 0.9 1.0 0.3\nEND\n' > "$T/c2dz.parms"
  if "$BIND" "$T/c2dz.bind" "$T/c2dz.parms" >/dev/null 2>&1 && grep -q "P DZ:" "$T/c2dz.bind.out"; then
    echo "  double-zeta p supported (basis_dz = X:p available)"
  else
    echo "  note: this bind has no double-zeta p; basis_dz = X:p will be refused (use --build-yaehmop)"
  fi
  # native spin polarization / SOC of the fork (spin_combine.py, soc_fit.py, spin_joint_fit.py)
  HELP="$("$BIND" --help 2>&1 || true)"
  if printf '%s' "$HELP" | grep -q -- "--parm-up" && printf '%s' "$HELP" | grep -q -- "--common-fermi" \
     && printf '%s' "$HELP" | grep -q -- "--soc"; then
    echo "  native spin polarization and SOC supported (--parm-up/--parm-down, --common-fermi, --soc)"
  else
    echo "  note: this bind has no native spin/SOC; spin_joint_fit.py needs it (use --build-yaehmop)"
  fi
  [ -s "$T/h2.bind.out" ] && rm -rf "$T"
else
  say "YAeHMOP not configured"
  echo "  Fitting needs it. Re-run with --build-yaehmop, or with --yaehmop-bind and --yaehmop-params,"
  echo "  or later: conda env config vars set -n $ENV_NAME YAEHMOP_BIND=... YAEHMOP_PARAM_FILE=..."
fi

if [ "$RUN_TESTS" = 1 ]; then
  say "tests"
  (cd "$HERE" && python -W ignore -m unittest discover -s tests -p 'test_*.py')
fi

say "done. Use:  conda activate $ENV_NAME"
echo "  then, from $HERE:  python quick.py   |   python USER_WORKFLOW/bands_fit.py --help"
