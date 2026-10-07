#!/bin/bash
#
#  quick tracking regression on the detected PRNs (golden results) for 1 s
#
#  usage: quick_trk.sh [IF_file [out_dir [backend ...]]]
#
#  The PRNs of L1CA, E1B and B1I detected in golden_cpu.txt (33 channels) are
#  acquired and tracked for TLEN (default 1) s by bench_trk.sh without the
#  scaling benchmark.
#
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
prns() {
    awk -v s=$1 '!/^#/ && NF >= 10 && $1 == s && $3 == 1 {print $2}' \
        golden_cpu.txt | paste -sd,
}
TRK_SIGS="-sig L1CA -prn $(prns L1CA) -rfch 1 -sig E1B -prn $(prns E1B) \
-rfch 1 -sig B1I -prn $(prns B1I) -rfch 2" TLEN=${TLEN:-1} SCALING=0 \
    ./bench_trk.sh "$@"
