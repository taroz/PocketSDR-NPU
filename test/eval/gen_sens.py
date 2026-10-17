#!/usr/bin/env python3
#
#  generate synthetic IF data for the acquisition sensitivity test
#  (Pocket SDR FE 4CH RAW16, 24 MHz, pocket_L1L2L5E5b_24MHz.conf)
#
#  Segments of TSEG s, each with L1CA, E1B and L1CP of PRN 1 in RF CH 1 (LO
#  1568 MHz) and L5I of PRN 1 in RF CH 3 (LO 1176.45 MHz) at a C/N0, random
#  code delay, Doppler (+-4 kHz, with code Doppler) and carrier
#  phase, without navigation data, white noise and 2-bit I/Q quantization (+-1,
#  +-3, threshold at the noise sigma). RF CH 2 and 4: noise only.
#
#  usage: gen_sens.py cn0 nseg out.bin [-noise]
#         (-noise: noise only) (truth: out.bin.txt, seg sig prn cn0 delay(ms)
#         doppler(Hz))
#
#  Requires numpy and python/sdr_code.py of Pocket SDR.
#
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../../python'))
import sdr_code

FS = 24e6           # sampling frequency (Hz)
TSEG = 0.05         # segment length (s)
TPAD = 0.2          # noise padding at the end (s) (IF data read by bench_sdr_acq)
FO = [1568e6, 1227.6e6, 1176.45e6, 1207.14e6] # LO frequencies of RF CHs (Hz)
SIGS = [('L1CA', 0), ('E1B', 0), ('L1CP', 0), ('L5I', 2)]
FREQ = {'L1CA': 1575.42e6, 'E1B': 1575.42e6, 'L1CP': 1575.42e6, 'L5I': 1176.45e6}
IDX_I = {1: 0, 3: 1, -1: 2, -3: 3} # 2-bit codes of I, Q (as read by bench_sdr_acq)
IDX_Q = {-1: 0, -3: 1, 1: 2, 3: 3}

# 2-bit quantization of complex samples to the codes of I, Q ------------------
def quant2(x, th):
    lut_i = np.array([IDX_I[v] for v in (-3, -1, 1, 3)], dtype=np.uint8)
    lut_q = np.array([IDX_Q[v] for v in (-3, -1, 1, 3)], dtype=np.uint8)
    q = lambda v: np.digitize(v, [-th, 0.0, th]) # 0: -3, 1: -1, 2: 1, 3: 3
    return lut_i[q(x.real)] | (lut_q[q(x.imag)] << 2)

# main ------------------------------------------------------------------------
def main(argv):
    cn0, nseg, out = float(argv[0]), int(argv[1]), argv[2]
    noise_only = '-noise' in argv[3:]
    rng = np.random.default_rng(int(cn0 * 10) + (1 if noise_only else 0))
    N = int(FS * TSEG)
    n = np.arange(N)
    codes = {s: np.array(sdr_code.gen_code(s, 1), dtype=np.float32) for s, _ in SIGS}
    th = np.sqrt(0.5) # noise sigma of I, Q
    with open(out, 'wb') as fp, open(out + '.txt', 'w') as ft:
        for g in range(nseg):
            x = [(rng.standard_normal(N) + 1j * rng.standard_normal(N)) * th
                for _ in range(4)]
            for sig, ch in SIGS:
                c = cn0
                T, code = sdr_code.code_cyc(sig), codes[sig]
                tau, fd = rng.uniform(0.0, T), rng.uniform(-4000.0, 4000.0)
                ph = rng.uniform(0.0, 1.0)
                if not noise_only:
                    t = n / FS * (1.0 + fd / FREQ[sig]) - tau # code time
                    ix = np.floor(t / T * len(code)).astype(np.int64) % len(code)
                    fi = FREQ[sig] - FO[ch] + fd
                    amp = np.sqrt(10 ** (c / 10) / FS)
                    x[ch] = x[ch] + amp * code[ix] * np.exp(2j * np.pi * (fi * n /
                        FS + ph))
                ft.write('%d %s 1 %.1f %.7f %.2f\n' % (g, sig, c if not noise_only
                    else -1.0, tau * 1e3, fd))
            b0 = quant2(x[0], th) | (quant2(x[1], th) << 4)
            b1 = quant2(x[2], th) | (quant2(x[3], th) << 4)
            fp.write(np.stack([b0, b1], axis=1).astype(np.uint8).tobytes())
        x = [(rng.standard_normal(int(FS * TPAD)) + 1j * rng.standard_normal(
            int(FS * TPAD))) * th for _ in range(4)]
        b0 = quant2(x[0], th) | (quant2(x[1], th) << 4)
        b1 = quant2(x[2], th) | (quant2(x[3], th) << 4)
        fp.write(np.stack([b0, b1], axis=1).astype(np.uint8).tobytes())
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: gen_sens.py cn0 nseg out.bin [-noise]')
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))
