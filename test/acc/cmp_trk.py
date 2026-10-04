#!/usr/bin/env python3
#
#  compare tracking results of bench_sdr_trk (reference vs test backend)
#
#  usage: cmp_trk.py ref.txt test.txt
#
#  Each line: sig prn ix lock coff fd adr phi cn0 sec_sync ssync fsync
#             C[0].re C[0].im ... C[n-1].re C[n-1].im P.re P.im
#
import sys
import math

TOL_CORR = 0.03     # median relative error of correlator outputs per epoch
TOL_DISC = 0.01     # max abs diff of DLL discriminator (E-L)/(E+L) (median of epochs)
TOL_COFF = 1e-9     # code offset (s) (1 ns = 0.3 m)
TOL_FD   = 1.0      # Doppler (Hz) (p99 of epochs, single epochs of FLL pull-in excluded)
TOL_ADR  = 0.05     # accumulated Doppler (cyc)
TOL_CN0  = 1.0      # C/N0 (dB-Hz)
TOL_LOCK = 0.99     # ratio of locked channel-epochs (test / ref)

# read results -----------------------------------------------------------------
def read_res(file):
    res = {}
    for line in open(file):
        v = line.split()
        key = (v[0], int(v[1]), int(v[2]))
        res[key] = (int(v[3]), [float(x) for x in v[4:9]],
            (int(v[9]), int(v[10]), int(v[11])), [float(x) for x in v[12:]])
    return res

# DLL discriminator from E/L correlators --------------------------------------
def disc(C):
    E = math.hypot(C[2], C[3])
    L = math.hypot(C[4], C[5])
    return (E - L) / (E + L) if E + L > 0 else 0.0

# main -------------------------------------------------------------------------
#   stable channels: locked in the same epochs of ref and test with mean C/N0
#   >= CN0_STABLE. The loss-of-lock decisions of weak channels (C/N0 or PLI near
#   the thresholds) may differ by any small numerical difference and the states
#   diverge chaotically after that, so they are counted as mismatches (allowed
#   up to MAX_MISMATCH of channels) and evaluated by the correlators of the
#   first epochs.
CN0_STABLE = 35.0   # C/N0 threshold of stable channels (dB-Hz)
N_FIRST    = 100    # first epochs of correlators of all channels
MAX_MISMATCH = 0.1  # max ratio of channels with different loss-of-lock

def main(argv):
    ref, tst = read_res(argv[1]), read_res(argv[2])
    err = []
    nref = len(ref)
    nboth = sum(1 for k in ref if k in tst)
    # channels
    chs = {}
    for k, r in ref.items():
        chs.setdefault(k[:2], []).append(k)
    tch = {}
    for k in tst:
        tch[k[:2]] = tch.get(k[:2], 0) + 1
    stable, mismatch = set(), []
    for ch, ks in chs.items():
        cn0 = sum(ref[k][1][4] for k in ks) / len(ks)
        if len(ks) != tch.get(ch, 0):
            mismatch.append('%s %d (%d->%d epochs)' % (ch[0], ch[1], len(ks),
                tch.get(ch, 0)))
        elif cn0 >= CN0_STABLE:
            stable.add(ch)
    for ch in tch:
        if ch not in chs: mismatch.append('%s %d (0->%d epochs)' % (ch[0],
            ch[1], tch[ch]))
    erel, efirst, ddisc, dfd = [], [], [], []
    md = {'coff': 0.0, 'fd': 0.0, 'adr': 0.0, 'cn0': 0.0}
    nst = nst_both = nsync = 0
    for ch, ks in chs.items():
        ks.sort(key=lambda k: k[2])
        for i, k in enumerate(ks):
            if k not in tst: continue
            r, t = ref[k], tst[k]
            num = sum((a - b) ** 2 for a, b in zip(r[3], t[3]))
            den = sum(a * a for a in r[3])
            e = math.sqrt(num / den) if den > 0 else 0.0
            if i < N_FIRST: efirst.append(e)
            if ch not in stable: continue
            erel.append(e)
            ddisc.append(abs(disc(r[3]) - disc(t[3])))
            for j, n in enumerate(('coff', 'fd', 'adr')):
                md[n] = max(md[n], abs(r[1][j] - t[1][j]))
            md['cn0'] = max(md['cn0'], abs(r[1][4] - t[1][4]))
            dfd.append(abs(r[1][1] - t[1][1]))
            nsync += r[2] == t[2]
        nst += len(ks) if ch in stable else 0
        nst_both += sum(1 for k in ks if k in tst) if ch in stable else 0
    pct = lambda v, p: sorted(v)[min(len(v) - 1, int(len(v) * p))] if v else 0.0
    bitexact = open(argv[1]).read() == open(argv[2]).read()
    print('channels=%d stable=%d channel-epochs ref=%d test=%d common=%d '
        'bit-exact=%s' % (len(chs), len(stable), nref, len(tst), nboth, bitexact))
    print('correlator rel. error: first %d epochs median=%.4f p95=%.4f, '
        'stable median=%.4f p95=%.4f' % (N_FIRST, pct(efirst, 0.5),
        pct(efirst, 0.95), pct(erel, 0.5), pct(erel, 0.95)))
    print('stable: DLL discriminator diff median=%.5f p99=%.5f, lock=%d/%d, '
        'sync-state same=%.4f' % (pct(ddisc, 0.5), pct(ddisc, 0.99), nst_both,
        nst, nsync / max(1, nst_both)))
    print('stable max diff: coff=%.3e s fd=%.4f Hz (p99=%.4f Hz) adr=%.4f cyc '
        'cn0=%.3f dB-Hz' % (md['coff'], md['fd'], pct(dfd, 0.99), md['adr'],
        md['cn0']))
    print('all channels: locked channel-epochs test/ref=%.4f, lock mismatch=%d: %s'
        % (len(tst) / nref, len(mismatch), ', '.join(mismatch)))
    if pct(efirst, 0.5) > TOL_CORR: err.append('correlators (first epochs)')
    if pct(erel, 0.5) > TOL_CORR: err.append('correlators (stable)')
    if pct(ddisc, 0.5) > TOL_DISC: err.append('discriminator')
    if len(mismatch) > MAX_MISMATCH * len(chs): err.append('lock mismatch')
    if md['coff'] > TOL_COFF: err.append('coff %.3e' % md['coff'])
    if pct(dfd, 0.99) > TOL_FD: err.append('fd %.3f' % pct(dfd, 0.99))
    if md['adr'] > TOL_ADR: err.append('adr %.3f' % md['adr'])
    if md['cn0'] > TOL_CN0: err.append('cn0 %.3f' % md['cn0'])
    if len(tst) / nref < TOL_LOCK: err.append('lock ratio')
    print('RESULT: %s %s' % ('PASS' if not err else 'FAIL', ', '.join(err)))
    return 0 if not err else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
