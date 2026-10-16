#!/bin/bash
#
#  acquisition time per signal by backends on recorded 24 MHz IF data of
#  pocket_L1L2L5E5b_24MHz.conf (bench_sdr_acq, all PRNs of the signals, the
#  narrowband signals decimated to 8 MHz, L5I at 24 MHz, L1CP and L2CM by the
#  direct search of the long codes)
#
#  usage: acq_bench.sh IF_file backend ...  (e.g. cpu rknn hailo:hef=<dir>)
#
#  environment: BENCH (default: ../utest/bench_sdr_acq), BENCH_OPTS (e.g.
#  "-wisdom fftw_wisdom.txt" for bench_sdr_acq_fftw with the wisdom of
#  wisdom_add.c), TASKSET (e.g. "taskset
#  -c 4"), SIGS (signal options), OUT (output directory, default: ./out)
#
EVAL=$(cd "$(dirname "$0")" && pwd)
F=$1; shift
B=${BENCH:-$EVAL/../utest/bench_sdr_acq}
OUT=${OUT:-$EVAL/out}; mkdir -p "$OUT"
export SDR_ACC_PATH=${SDR_ACC_PATH:-$EVAL/../../lib/linux}
C="-fmt RAW16 -f 24 -fo 1568,1227.6,1176.45,1207.14 -toff 10 -dec 1:3:7.42,1:3:-6.902,2:3"
SIGS=${SIGS:-"-sig L1CA -prn 1-32 -rfch 1 -sig E1B -prn 1-36 -rfch 1 -sig B1I -prn 1-63 -rfch 1 \
-sig L5I -prn 1-32 -rfch 3 -sig L1CP -prn 1-32 -rfch 1 -sig L2CM -prn 1-32 -rfch 2"}
for a in "$@"; do
    n=$(echo "$a" | cut -d: -f1)
    echo "=== $a"
    $TASKSET $B $C $BENCH_OPTS -acc $a $SIGS -o "$OUT/acq_$n.txt" "$F" > "$OUT/acq_$n.log" 2>&1
    grep -E "^# [A-Z0-9]+:|fallback" "$OUT/acq_$n.log" | sed -E 's/ in=[0-9.]+ out=[0-9.]+ kB\/call//'
done
