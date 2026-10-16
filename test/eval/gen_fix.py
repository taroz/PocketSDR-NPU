#!/usr/bin/env python3
# generate fixed-set capacity signal files: the first N of a stratified order
# of the reliable signals (tracked without loss to the end in the cpu_opt
# reference runs), nested sets (S_N1 in S_N2 for N1 < N2), N > |S| by
# duplicating the signals in the same order, + background search channels
# (signals never found, the same as gen_cap.py)
# usage: gen_fix.py reliable.txt found_all.txt N out.sh [-nobg]
# (-nobg: no background search channels)
# (fix_<N>.sh: gen_fix.py reliable24.txt found24.txt N fix_<N>.sh, reliable24:
# the signals tracked to the end of the 24 MHz reference runs, found24: the
# signals of the background list ever found)
import sys, random
RFCH = {'L1CA':1,'L1CD':1,'L1CP':1,'E1B':1,'E1C':1,'B1CD':1,'B1CP':1,'B1I':1,
    'L1S':1,'I1SD':1,'I1SP':1,'L2CM':2,'L5I':3,'L5Q':3,'E5AI':3,'E5AQ':3,
    'B2AD':3,'B2AP':3,'I5S':3,'L5SI':3,'L5SQ':3,'E5BI':4,'E5BQ':4,'B2I':4,'B2BI':4}
BG = [('L1CA',list(range(1,33))+list(range(120,159))+list(range(193,203))),
    ('E1B',range(1,37)),('B1I',range(1,64)),('I5S',range(1,15))]
S = [tuple(l.strip().split(':')) for l in open(sys.argv[1]) if l.strip()]
found = {tuple(l.strip().split(':')) for l in open(sys.argv[2]) if l.strip()}
N = int(sys.argv[3])
random.seed(1)
by = {}
for s, p in sorted(S): by.setdefault(s, []).append(p)
for v in by.values(): random.shuffle(v)
# stratified order: the i-th signal of a type at position (i + 0.5) / n_type
order = sorted(((i + 0.5) / len(v), s, p) for s, v in by.items() for i, p in enumerate(v))
order = [(s, p) for _, s, p in order]
sel = (order * (N // len(order) + 1))[:N]
def opts(pairs):
    g = {}
    for s, p in pairs: g.setdefault(s, []).append(p)
    return ' '.join('-sig %s -prn %s -rfch %d' % (s, ','.join(v), RFCH[s]) for s, v in g.items())
bg = [(s, str(p)) for s, r in BG for p in r if (s, str(p)) not in found]
if '-nobg' in sys.argv[5:]: bg = []
with open(sys.argv[4], 'w') as f:
    f.write('FO=1568,1227.6,1176.45,1207.14\n')
    f.write('SIGS="%s %s"\n' % (opts(sel), opts(bg)))
n24 = sum(RFCH[s] >= 3 for s, p in sel)
print('N=%d (%d unique, %d at 24 MHz), %d background' % (len(sel), len(set(sel)), n24, len(bg)))
