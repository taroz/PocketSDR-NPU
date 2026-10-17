#!/usr/bin/env python3
#
#  summary of the acquisition sensitivity test: detection probability (found
#  with the code delay within +-1.5 samples and the Doppler within a bin step
#  of the truth), false alarm probability (noise only), C/N0 metric and errors
#
#  usage: sens_sum.py truth_dir dir ...  (dir: res_<cn0>.txt of bench_sdr_acq -nseg,
#         the truth s<cn0>.bin.txt of gen_sens.py, res_noise.txt for noise)
#
import sys, os, re, glob, statistics as st

FS = {'L1CA': 8e6, 'E1B': 8e6, 'L1CP': 8e6, 'L5I': 24e6} # (after decimation)
T = {'L1CA': 1e-3, 'E1B': 4e-3, 'L1CP': 10e-3, 'L5I': 1e-3}

def read(file):
    return {(v[0], int(v[10])): v for v in (l.split() for l in open(file)) if len(v) >= 11}

def summary(d, tdir):
    rows = {}
    files = sorted(glob.glob(d + '/res_[0-9]*.txt'), key=lambda f: float(re.findall(r'res_(\d+(?:\.\d+)?)\.txt', f)[0]))
    # code delay offset of decimation by signal (median of detections at max C/N0)
    res_max = read(files[-1]); tru_max = truth(tdir, files[-1])
    off = {}
    for s in FS:
        v = [dcode(res_max[k], tru_max[k]) for k in res_max if k[0] == s and res_max[k][2] == '1']
        off[s] = st.median(v) if v else 0.0
    for f in files:
        cn0 = float(re.findall(r'res_(\d+(?:\.\d+)?)\.txt', f)[0])
        res, tru = read(f), truth(tdir, f)
        for s in FS:
            ks = [k for k in res if k[0] == s]
            if not ks: continue
            ok = [k for k in ks if res[k][2] == '1' and abs(dcode(res[k], tru[k]) - off[s]) <= 1.5
                and abs(float(res[k][6]) - tru[k][1]) <= 0.5 / T[s]]
            m = [float(res[k][3]) for k in ks]
            rows.setdefault(s, []).append((cn0, len(ok) / len(ks), len(ks), st.mean(m)))
    nf = glob.glob(d + '/res_noise.txt')
    pfa = {}
    if nf:
        res = read(nf[0])
        for s in FS:
            ks = [k for k in res if k[0] == s]
            if ks: pfa[s] = (sum(res[k][2] == '1' for k in ks), len(ks))
    for s in rows:
        print('%s %s (Pfa %s)' % (d, s, '%d/%d' % pfa[s] if s in pfa else '-'))
        for cn0, pd, n, m in rows[s]:
            print('  C/N0 %4.1f: Pd %5.3f (n=%3d) C/N0 metric %5.1f' % (cn0, pd, n, m))

def truth(tdir, f):
    cn0 = re.findall(r'res_(\d+(?:\.\d+)?)\.txt', f)[0]
    return {(v[1], int(v[0])): (float(v[4]), float(v[5])) for v in
        (l.split() for l in open(os.path.join(tdir, 's%s.bin.txt' % cn0)))}

def dcode(r, t): # code delay (result - truth) (samples) modulo code period
    s = r[0]; n = FS[s] * T[s]
    d = (float(r[7]) - t[0]) * 1e-3 * FS[s]
    return (d + n / 2) % n - n / 2

if __name__ == '__main__':
    tdir = sys.argv[1]
    for d in sys.argv[2:]:
        summary(d, tdir)
