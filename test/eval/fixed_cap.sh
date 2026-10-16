#!/bin/bash
#
#  fixed-set capacity: the same tracked signals (fix_<N>.sh, the fixed set of N
#  signals + background search channels) by two acquisition backends, runs
#  alternated, 300 s of IF data, real-time input (-rcv_prio), no new search
#  while the tracking is delayed (-trk_guard 20), decimation of the narrowband
#  sub-bands of the 24 MHz IF data of pocket_L1L2L5E5b_24MHz.conf
#
#  usage: fixed_cap.sh "N ..." reps backend_a backend_b
#         (e.g. fixed_cap.sh "140 157 175 200" 2 hailo:hef=<dir>,batch=16 cpu)
#
#  environment: INPUT (300 s IF data), TRK, OUT, CAP_OPTS (other options, e.g.
#  "-opt fftw.conf" with FFTW wisdom) (see e2e.sh), the results summarized by
#  cap_sum.py OUT/fx_*
#
#  -rcv_prio needs CAP_SYS_NICE (e.g. sudo setcap cap_sys_nice+ep pocket_trk).
#
EVAL=$(cd "$(dirname "$0")" && pwd)
export TLIM=${TLIM:-305}
DEC="1:3:7.42,1:3:-6.902,2:3 -LOG_LEVEL=4" # (log level 4: buffer usage)
tag() { echo "$1" | cut -d: -f1; }
run() {
    SIGFILE=$EVAL/fix_$2.sh $EVAL/e2e.sh fx_$2_$(tag $3)_$1 -acc $3 -rcv_prio 10 \
        -trk_guard 20 -dec "$DEC" $CAP_OPTS
    sleep 30
}
for r in $(seq 1 ${2:-1}); do
    for N in $1; do
        if [ $((r % 2)) = 1 ]; then run $r $N "$3"; run $r $N "$4"
        else run $r $N "$4"; run $r $N "$3"; fi
    done
done
