#!/usr/bin/env python3
#
#  compare acquisition results of bench_sdr_acq (reference vs test backend)
#
#  usage: cmp_acq.py ref.txt test.txt [-thres 34.0] [-margin 0.5]
#
import sys

TOL_CN0  = 0.5    # C/N0 metric (dB)
TOL_COFF = 1      # code phase (samples)
TOL_FD   = 0.1    # fine Doppler (x Doppler bin step)
TOL_PMAX = 0.05   # peak correlation power (relative)

# read results file ------------------------------------------------------------
def read_res(file):
    res = {}
    for line in open(file):
        v = line.split()
        if len(v) < 10: continue
        res[(v[0], int(v[1]))] = {'found': int(v[2]), 'cn0': float(v[3]),
            'idx': (int(v[4]), int(v[5])), 'fd': float(v[6]),
            'coff': float(v[7]), 'pmax': float(v[8]), 'hash': v[9]}
    return res

# code cycle (ms) of signal ----------------------------------------------------
def code_cyc(sig):
    cyc = {'E1B': 4.0, 'E1C': 4.0, 'L6D': 4.0, 'L6E': 4.0, 'L1CP': 10.0,
        'L1CD': 10.0, 'B1CP': 10.0, 'B1CD': 10.0, 'L2CM': 20.0}
    return cyc.get(sig, 1.0)

# main -------------------------------------------------------------------------
def main(argv):
    thres, margin, files = 34.0, 0.5, []
    i = 1
    while i < len(argv):
        if argv[i] == '-thres': thres = float(argv[i+1]); i += 1
        elif argv[i] == '-margin': margin = float(argv[i+1]); i += 1
        else: files.append(argv[i])
        i += 1
    ref, tst = read_res(files[0]), read_res(files[1])
    nerr = nsame = nhash = nborder = ndet = 0
    max_d = {'cn0': 0.0, 'coff': 0, 'fd': 0.0, 'pmax': 0.0}
    for key, r in ref.items():
        if key not in tst:
            print('missing %s %d' % key); nerr += 1; continue
        t = tst[key]
        nhash += r['hash'] == t['hash']
        err = []
        border = abs(r['cn0'] - thres) < margin
        if r['found'] != t['found']:
            if border: nborder += 1
            else: err.append('detection %d->%d' % (r['found'], t['found']))
        else:
            nsame += 1
        dcn0 = abs(r['cn0'] - t['cn0'])
        max_d['cn0'] = max(max_d['cn0'], dcn0)
        if dcn0 > TOL_CN0: err.append('cn0 %.2f' % dcn0)
        if r['found']:
            ndet += 1
            dcoff = abs(r['idx'][1] - t['idx'][1])
            step = 500.0 / code_cyc(key[0])
            dfd = abs(r['fd'] - t['fd']) / step
            dp = abs(r['pmax'] - t['pmax']) / r['pmax']
            max_d['coff'] = max(max_d['coff'], dcoff)
            max_d['fd'] = max(max_d['fd'], dfd)
            max_d['pmax'] = max(max_d['pmax'], dp)
            if dcoff > TOL_COFF: err.append('code phase %d' % dcoff)
            # adjacent coarse bin allowed if fine Doppler agrees (equal powers)
            if abs(r['idx'][0] - t['idx'][0]) > 1: err.append('Doppler bin')
            if dfd > TOL_FD: err.append('fine Doppler %.3f bin' % dfd)
            if dp > TOL_PMAX: err.append('peak %.3f' % dp)
        if err:
            nerr += 1
            print('NG %-6s %3d: %s' % (key[0], key[1], ', '.join(err)))
    print('signals=%d detected(ref)=%d same-detection=%d borderline=%d '
        'bit-exact=%d' % (len(ref), ndet, nsame, nborder, nhash))
    print('max diff: cn0=%.3f dB code=%d sample fd=%.4f bin peak=%.4f' % (
        max_d['cn0'], max_d['coff'], max_d['fd'], max_d['pmax']))
    print('RESULT: %s (%d errors)' % ('PASS' if nerr == 0 else 'FAIL', nerr))
    return 0 if nerr == 0 else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
