#!/bin/bash
#
#  quick acquisition regression and benchmark on a representative subset
#
#  usage: quick_acq.sh [IF_file [out_dir [backend ...]]]
#
#  For each signal of acq_sigs.sh, the strongest and the weakest detected PRNs
#  and the strongest undetected PRN of the golden results are selected (30
#  signals), and bench_acq.sh runs them and compares with the golden subset.
#
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
FILE=${1:-$HOME/IFdata/L1B1G1L2G2L5E5bL6_20250327_0248_16MHz.bin}
OUT=${2:-./out_quick}
shift 2 2>/dev/null
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
source ./acq_sigs.sh
python3 - "$SIGS" golden_cpu.txt "$OUT/quick_sigs.sh" "$OUT/golden_quick.txt" <<'PY' || exit 1
import sys
# select PRNs of each signal from the golden results
sigs, golden, sigfile, subfile = sys.argv[1:5]
rows = [l.split() for l in open(golden) if len(l.split()) >= 10]
v, opt, sel = sigs.split(), [], set()
for i in range(0, len(v), 6):
    sig, rfch = v[i+1], v[i+5]
    r = [x for x in rows if x[0] == sig]
    det = sorted([x for x in r if x[2] == '1'], key=lambda x: float(x[3]))
    und = sorted([x for x in r if x[2] == '0'], key=lambda x: float(x[3]))
    prns = {x[1] for x in (det[-1:] + det[:1] + und[-1:])}
    sel |= {(sig, p) for p in prns}
    opt.append('-sig %s -prn %s -rfch %s' % (sig,
        ','.join(sorted(prns, key=int)), rfch))
open(sigfile, 'w').write('source ./acq_sigs.sh\nSIGS="%s"\n' %
    ' '.join(opt))
with open(subfile, 'w') as f:
    for x in rows:
        if (x[0], x[1]) in sel: f.write(' '.join(x) + '\n')
PY
SIGFILE="$OUT/quick_sigs.sh" REF="$OUT/golden_quick.txt" \
    ./bench_acq.sh "$FILE" "$OUT" "$@"
