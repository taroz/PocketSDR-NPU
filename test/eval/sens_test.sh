#!/bin/bash
#
#  acquisition sensitivity test: bench_sdr_acq on the synthetic IF data of
#  gen_sens.py (s<cn0>.bin, s_noise.bin in DATA) by backends
#
#  usage: sens_test.sh DATA OUT backend ...
#
#  environment: BENCH (default: ../utest/bench_sdr_acq), BENCH_OPTS, TASKSET,
#  NSEG (segments per C/N0, default: 100), NSEG_L (L1CP, default: 50)
#
EVAL=$(cd "$(dirname "$0")" && pwd)
DATA=$1; OUT=$2; shift 2
B=${BENCH:-$EVAL/../utest/bench_sdr_acq}
export SDR_ACC_PATH=${SDR_ACC_PATH:-$EVAL/../../lib/linux}
C="-fmt RAW16 -f 24 -fo 1568,1227.6,1176.45,1207.14 -toff 0 -tseg 0.05 -dec 1:3:7.42"
for a in "$@"; do
    n=$(echo "$a" | cut -d: -f1); mkdir -p $OUT/$n
    for f in $DATA/s*.bin; do
        c=$(basename $f .bin); c=${c#s}; c=${c#_}
        $TASKSET $B $C $BENCH_OPTS -acc $a -nseg ${NSEG:-100} -sig L1CA -prn 1 -rfch 1 \
            -sig E1B -prn 1 -rfch 1 -sig L5I -prn 1 -rfch 3 -o $OUT/$n/r1.txt $f > /dev/null 2>&1
        $TASKSET $B $C $BENCH_OPTS -acc $a -nseg ${NSEG_L:-50} -sig L1CP -prn 1 -rfch 1 \
            -o $OUT/$n/r2.txt $f > /dev/null 2>&1
        cat $OUT/$n/r1.txt $OUT/$n/r2.txt > $OUT/$n/res_$c.txt
        echo "$a $c found $(awk '$3 == 1' $OUT/$n/res_$c.txt | wc -l)/$(wc -l < $OUT/$n/res_$c.txt)"
    done
done
