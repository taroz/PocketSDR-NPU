#!/usr/bin/env python3
#
#  summary of fixed-set capacity runs (fixed_cap.sh): the channels 1..N are the
#  fixed set of fix_<N>.sh (N from the directory name fx_<N>_...)
#
#  TRK : max / mean of the IF buffer usage of the tracking channels (%) and the
#        seconds > 100% (overflow, not real time) (receiver log level 4)
#  lock: fixed channels locked by 10 / 30 / 60 / 300 s, time to 90% / 95% of
#        the fixed set (s)
#  loss: gaps > 2 s of the channel log of the fixed channels
#  CPU : mean CPU usage per core 30-300 s (mon.txt)
#
#  usage: cap_sum.py dir ...
#
import sys, re

TEND = 300.0

def summary(d):
    N = int(re.search(r'fx_(\d+)_', d)[1])
    first, last, loss, trk = {}, {}, 0, []
    for l in open(d + '/trk.log', errors='ignore'):
        if l.startswith('$CH,'):
            f = l.split(',')
            t, ch = float(f[1]), int(f[2])
            if t > TEND or ch > N: continue
            if ch not in first: first[ch] = t
            elif t - last[ch] > 2.0: loss += 1
            last[ch] = t
        elif l.startswith('$LOG,') and ',BUFF ' in l:
            m = re.match(r'\$LOG,([\d.]+),,0,BUFF ([\d.]+) TRK ([\d.]+)', l)
            if m and float(m[1]) <= TEND: trk.append(float(m[3]))
    ts = sorted(first.values())
    at = lambda T: sum(t <= T for t in ts)
    tq = lambda q: ts[int(q * N) - 1] if len(ts) >= int(q * N) else float('nan')
    cpu = []
    for l in open(d + '/mon.txt'):
        f = l.split('|')
        if len(f) > 1 and 30 <= int(f[0]) <= TEND:
            v = [int(x) for x in f[1].split()[1:]]
            cpu.append(sum(v) / len(v))
    trk = trk or [float('nan')]
    print('%-20s TRK %5.1f %4.1f %3d s | lock %3d %3d %3d %3d /%d t90 %5.1f t95 %5.1f | '
        'loss %3d | CPU %3.0f%%' % (d, max(trk), sum(trk) / len(trk),
        sum(x > 100 for x in trk), at(10), at(30), at(60), at(TEND), N, tq(0.9),
        tq(0.95), loss, sum(cpu) / max(len(cpu), 1)))

if __name__ == '__main__':
    for d in sys.argv[1:]:
        summary(d)
