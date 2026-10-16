#!/bin/bash
#
#  pocket_trk on recorded IF data with a 10 s monitor of the CPU usage and the
#  receiver status (mon.txt)
#
#  usage: e2e.sh name [pocket_trk options ...]
#
#  environment:
#    INPUT  : IF data file (with .tag)
#    SIGFILE: signal file (FO=..., SIGS="...", e.g. fix_157.sh)
#    TLIM   : time limit (s) (SIGINT)
#    TRK    : pocket_trk (default: ../../app/pocket_trk/pocket_trk)
#    OUT    : output directory (default: ./out), the results in OUT/name
#
EVAL=$(cd "$(dirname "$0")" && pwd)
name=$1; shift
D=${OUT:-$EVAL/out}/$name; mkdir -p "$D"; cd "$D" || exit 1
source "${SIGFILE:?SIGFILE not set}"
export SDR_ACC_PATH=${SDR_ACC_PATH:-$EVAL/../../lib/linux}
echo "$@" > opts.txt
NCPU=$(nproc)
t0=$(date +%s)
( time ${TLIM:+timeout -s INT $TLIM} ${TRK:-$EVAL/../../app/pocket_trk/pocket_trk} \
    -fo $FO $SIGS "$@" -log trk.log -nmea sol.nmea "${INPUT:?INPUT not set}" \
    > out.txt 2> err.txt ) 2> time.txt &
pid=$!
cpus() { grep "^cpu[0-9]* " /proc/stat | head -$((NCPU + 1)) | tail -$NCPU | awk '{print $2+$3+$4+$7+$8}' | tr "\n" " "; }
prev=$(cpus)
while kill -0 $pid 2>/dev/null; do
    sleep 10
    cur=$(cpus)
    use=$(paste <(echo $prev | tr " " "\n") <(echo $cur | tr " " "\n") | awk '{printf "%3d ", ($2-$1)/10}')
    prev=$cur
    st=$(tail -c 30000 out.txt | tr "\r" "\n" | grep -a "BUFF:" | tail -1 | sed "s/\x1b\[[0-9;?]*[A-Za-z]//g" | tr -s " ")
    tmp=""
    if command -v vcgencmd > /dev/null; then # Raspberry Pi
        tmp="temp $(vcgencmd measure_temp | cut -d= -f2) $(vcgencmd get_throttled | cut -d= -f2) $(( $(vcgencmd measure_clock arm | cut -d= -f2) / 1000000 ))MHz"
    fi
    echo "$(( $(date +%s) - t0 )) | cpu% $use| $tmp | $st" >> mon.txt
done
