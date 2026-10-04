#!/bin/bash
#
#  tracking correlator backend regression and benchmark on recorded IF data
#
#  usage: bench_trk.sh [IF_file [out_dir [backend ...]]]
#
#  The signals (L1CA, E1B, B1I) are acquired and tracked for 3 s by the
#  original channel update (reference) and by the batched tracking correlators
#  of each backend, and the per-epoch tracking states are compared
#  (cmp_trk.py). The scaling of the batched correlators by channels and taps
#  is measured by bench_sdr_corr.
#  Set TASKSET (e.g. "taskset -c 12-15") to run on specific CPU cores.
#
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
FILE=${1:-$HOME/IFdata/L1B1G1L2G2L5E5bL6_20250327_0248_16MHz.bin}
OUT=${2:-./out_trk}
shift 2 2>/dev/null
BACKENDS=${@:-cpu cpu_mt npu}
BENCH=../utest/bench_sdr_trk
export SDR_ACC_PATH=${SDR_ACC_PATH:-$SCRIPT_DIR/../../lib/linux}
ARGS="-f 16 -fo 1575.42,1561,1602,1227.6,1246,1176.45,1207.14,1273 -tlen 3 \
-sig L1CA -prn 1-32 -rfch 1 -sig E1B -prn 1-36 -rfch 1 -sig B1I -prn 1-63 -rfch 2"

mkdir -p "$OUT"
make -C ../utest bench_sdr_trk bench_sdr_corr > /dev/null || exit 1
echo "=== reference (original channel update)"
$TASKSET $BENCH $ARGS -o "$OUT/trk_ref.txt" "$FILE" 2>&1 | grep '^#'
for be in $BACKENDS; do
    name=${be//[:,]/_}
    echo "=== batched: $be"
    $TASKSET $BENCH $ARGS -batch -tacc $be -o "$OUT/trk_$name.txt" "$FILE" 2>&1 |
        grep '^#'
    python3 cmp_trk.py "$OUT/trk_ref.txt" "$OUT/trk_$name.txt"
done
echo "=== scaling (channels x taps)"
BL=$(echo $BACKENDS | tr ' ' ',')
$TASKSET ../utest/bench_sdr_corr -nch 16,32,64,128,256,512 -ntap 4 -tacc $BL 2>&1 |
    grep -v compiled
$TASKSET ../utest/bench_sdr_corr -nch 64,256 -ntap 11,21,51,105 -tacc $BL -n 10 2>&1 |
    grep -v compiled
