#!/usr/bin/env python3
#
#  generate INT8 RKNN models of parallel code search for Rockchip NPU plugin
#  (libsdr_acc_rknn.so)
#
#  usage: gen_rknn_acq.py [-o dir] [-t target] [-fs fs] M:Kc[:R] ...
#
#    M  : DFT size (2 x samples per code period)
#    Kc : Doppler bins per inference (= ceil(K / NPU cores), K: bins of search)
#    R  : output rows of IDFT (default: A / 2, A x B = M, nout = R * B)
#    -o : output directory (default: ../linux/rknn)
#    -t : target platform (default: rk3588)
#    -fs: sampling frequency (MHz) of the calibration signals (default: 16)
#
#  The model (acq_M<M>_K<Kc>_R<R>.rknn) is the four-step DFT of the code
#  search in the NCHW layout of the NPU (each DFT stage a 1x1 Conv, complex by
#  3 real Conv, the axis changes 2 Transposes):
#
#    xr, xi [1][n1][1][n2] (IF samples wiped by the carrier of the first bin)
#      -> F_A -> twiddle -> Transpose -> F_B (forward DFT shared by the bins)
#      -> * code (cr, ci: code DFT rolled by bin, |IDFT(X[m+k] C[m])| =
#         |IDFT(X[m] C[m-k])|, Doppler bins spaced by fs / M)
#      -> F_B^-1 -> twiddle -> Transpose -> F_A^-1
#      -> P [Kc][p1][1][p2] = |c[B*p1+p2]|^2 / 64 (p1 < R)
#
#  All ops are INT8 except the last IDFT stage F_A^-1 and the correlation
#  power (FP16) (hybrid quantization), calibrated by synthetic signals (C/N0
#  30-60 dB-Hz) of the codes of the period M / 2 / fs, the code DFT normalized
#  by max magnitude.
#
#  e.g. gen_rknn_acq.py -fs 8 16000:7 64000:27 160000:34:200 320000:27:250
#       (8 MHz: 1 ms, 4 ms, L1CP 10 ms and L2CM 20 ms codes, 3 cores)
#
#  Requires RKNN-Toolkit2 (same version as the RKNN runtime of the target),
#  onnx, onnxruntime and numpy.
#
import sys, os
import numpy as np

MIN_FACT = 8
MAX_W = 8192    # max width of NPU tensors (larger: bins in batch)

# factorize DFT size (same as sdr_acc_rknn.cpp) -------------------------------
def factorize(M):
    for a in range(int(np.sqrt(M)), MIN_FACT - 1, -1):
        if M % a == 0 and (M // a) % 4 == 0: # B % 4 = 0 (as the plugin)
            return a, M // a
    return 0, 0

# ONNX graph ------------------------------------------------------------------
def build_onnx_shift(A, B, Kc, R):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    M, W = A * B, Kc * A
    flat = W <= MAX_W   # bins in width (batch 1)
    nodes, inits, cnt = [], [], [0]
    
    def nm(p):
        cnt[0] += 1
        return '%s_%d' % (p, cnt[0])
    
    def const(v, dtype=np.float32):
        name = nm('c')
        inits.append(nh.from_array(np.asarray(v, dtype=dtype), name))
        return name
    
    def op(t, ins, **kw):
        o = nm(t)
        nodes.append(h.make_node(t, ins, [o], **kw))
        return o
    
    def dft(n, sgn, rows):  # w[k][n] (out, in) unitary
        k = np.asarray(rows)[:, None] * np.arange(n)[None, :]
        return np.exp(sgn * 2j * np.pi * (k % n) / n) / np.sqrt(n)
    
    def cconv(xr, xi, w):  # complex 1x1 Conv by 3 real Conv
        c1 = op('Conv', [op('Add', [xr, xi]), const(w.real[:, :, None, None])])
        yr = op('Add', [op('Conv', [xi, const(-(w.real + w.imag)[:, :, None, None])]), c1])
        yi = op('Add', [op('Conv', [xr, const((w.imag - w.real)[:, :, None, None])]), c1])
        return yr, yi
    
    def cmul(ar, ai, br, bi):
        r = op('Sub', [op('Mul', [ar, br]), op('Mul', [ai, bi])])
        i = op('Add', [op('Mul', [ar, bi]), op('Mul', [ai, br])])
        return r, i
    
    def reshape(x, shape):
        return op('Reshape', [x, const(shape, np.int64)])
    
    cs = [1, B, 1, W] if flat else [Kc, B, 1, A]
    ins = [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in
        [('xr', [1, A, 1, B]), ('xi', [1, A, 1, B]), ('cr', cs), ('ci', cs)]]
    k1 = np.arange(A)[:, None]
    n2 = np.arange(B)[None, :]
    t1 = np.exp(-2j * np.pi * ((k1 * n2) % M) / M)[None, :, None, :] # [1][k1][1][n2]
    t2 = np.exp(2j * np.pi * ((n2.T * k1.T) % M) / M)              # [p2][k1]
    perm = [0, 3, 2, 1]
    
    # forward DFT once (shared by the bins)
    yr, yi = cconv('xr', 'xi', dft(A, -1, range(A)))    # [1][k1][1][n2]
    yr, yi = cmul(yr, yi, const(t1.real), const(t1.imag))
    zr, zi = op('Transpose', [yr], perm=perm), op('Transpose', [yi], perm=perm)
    ur, ui = cconv(zr, zi, dft(B, -1, range(B)))        # [1][k2][1][k1]
    # replicate for the bins by Concat (Tile and broadcast Mul gave wrong
    # results on the RK3588 NPU, correct in the simulator)
    ax = 3 if flat else 0
    ur, ui = (op('Concat', [ur] * Kc, axis=ax), op('Concat', [ui] * Kc, axis=ax))
    if flat:
        t2 = np.tile(t2, (1, Kc))                       # [1][k2][1][(k,k1)]
    vr, vi = cmul(ur, ui, 'cr', 'ci')                   # (or [Kc][k2][1][k1])
    qr, qi = cconv(vr, vi, dft(B, 1, range(B)))         # [.][p2][1][.]
    t2 = t2[None, :, None, :]
    sr, si = cmul(qr, qi, const(t2.real), const(t2.imag))
    sr, si = op('Transpose', [sr], perm=perm), op('Transpose', [si], perm=perm)
    if flat:
        sr, si = reshape(sr, [Kc, A, 1, B]), reshape(si, [Kc, A, 1, B])
    nodes[-2].output[0], nodes[-1].output[0] = 'Sr', 'Si' # (for hybrid quant.)
    orr, oi = cconv('Sr', 'Si', dft(A, 1, range(R)) / 8.0)  # [Kc][p1][1][p2]
    nodes[-3].output[0] = 'Or'  # (named for hybrid quantization)
    nodes[-1].output[0] = 'Oi'
    op('Add', [op('Mul', ['Or', 'Or']), op('Mul', ['Oi', 'Oi'])])
    nodes[-1].output[0] = 'P'
    outs = [h.make_tensor_value_info('P', T.FLOAT, [Kc, R, 1, B])]
    g = h.make_graph(nodes, 'sdr_acq_s', ins, outs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

# check ONNX graph of shared DFT by FFT ---------------------------------------
def check_onnx_shift(file, A, B, Kc, R):
    import onnxruntime as ort
    M, W = A * B, Kc * A
    rng = np.random.default_rng(1)
    x = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    f0, n = 0.3141, np.arange(M)  # carrier of first bin (cyc/sample)
    xw = (x * np.exp(-2j * np.pi * f0 * n)).reshape(1, A, 1, B)
    cr = [np.roll(c, k).reshape(B, A) for k in range(Kc)]   # c[m - k]
    cs = (np.concatenate(cr, axis=1)[None, :, None, :] if W <= MAX_W else
        np.stack(cr)[:, :, None, :])
    o = ort.InferenceSession(file).run(None, {k: v.astype(np.float32) for k, v in
        {'xr': xw.real, 'xi': xw.imag, 'cr': cs.real, 'ci': cs.imag}.items()})[0]
    ref = np.array([np.abs(np.fft.ifft(np.fft.fft(x * np.exp(-2j * np.pi *
        (f0 + k / M) * n)) * c)) ** 2 / 64 for k in range(Kc)])[:, :R * B]
    got = o.reshape(Kc, R * B)
    return np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref ** 2))

# calibration dataset for INT8 model ------------------------------------------
#   synthetic IF samples (noise + signal of random code, C/N0, code offset and
#   Doppler) wiped by the carrier of the first bin, normalized to unit RMS, and
#   the code DFT normalized by max magnitude (same as the plugin)
CAL_SIGS = ['L1CA', 'B1I', 'G1CA', 'L5I', 'E5AI', 'E6B', 'E1B', 'L6D', 'L1CP',
    'L2CM', 'E5BI', 'B2AD', 'B3I']
CAL_PRN = {'G1CA': list(range(-7, 7)), 'G2CA': list(range(-7, 7)),
    'L6D': list(range(193, 203)), 'L6E': list(range(203, 212))}

def gen_calib(A, B, Kc, fs, ncal, dir):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'python'))
    import sdr_code
    M, N = A * B, A * B // 2
    T = N / fs
    sigs = [s for s in CAL_SIGS if abs(sdr_code.code_cyc(s) - T) < 1e-9]
    if not sigs:
        raise ValueError('no code for period %.3f ms' % (T * 1e3))
    rng = np.random.default_rng(1)
    os.makedirs(dir, exist_ok=True)
    lines = []
    for i in range(ncal):
        sig = sigs[i % len(sigs)]
        prn = int(rng.choice(CAL_PRN.get(sig, list(range(1, 33)))))
        code = sdr_code.gen_code(sig, prn)
        while len(code) == 0:
            prn = int(rng.choice(CAL_PRN.get(sig, list(range(1, 33)))))
            code = sdr_code.gen_code(sig, prn)
        cn0 = rng.uniform(30.0, CAL_CN0) if i % 4 else 0.0 # 1/4 noise only
        amp = np.sqrt(10 ** (cn0 / 10) / fs) if cn0 > 0 else 0.0
        coff = rng.uniform(0, T)
        fd = rng.uniform(-1.0, Kc + 1.0) * fs / M     # vs first bin
        n = np.arange(M)
        rc = sdr_code.res_code(code, T, coff, fs, M)
        x = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
        x = x + amp * rc * np.exp(2j * np.pi * fd * n / fs)
        x /= np.sqrt(np.mean(np.abs(x) ** 2))
        c = sdr_code.gen_code_fft(code, T, 0.0, fs, N, N)
        c /= np.max(np.abs(c))
        xs = x.reshape(1, A, 1, B)
        # code DFT rolled by bin: C[(k1+A*k2-k) mod M]
        cs = np.stack([np.roll(c, k).reshape(B, A) for k in range(Kc)])
        if Kc * A <= MAX_W:
            cs = cs.transpose(1, 0, 2).reshape(1, B, 1, Kc * A)
        else:
            cs = cs[:, :, None, :]
        fs_ = []
        for k, v in [('xr', xs.real), ('xi', xs.imag), ('cr', cs.real), ('ci', cs.imag)]:
            f = os.path.join(dir, '%s_%d.npy' % (k, i))
            np.save(f, v.astype(np.float32))
            fs_.append(f)
        lines.append(' '.join(fs_))
    file = os.path.join(dir, 'dataset.txt')
    open(file, 'w').write('\n'.join(lines) + '\n')
    return file

# build INT8 model (hybrid: correlation power in FP16) ------------------------
NCAL = int(os.environ.get('RKNN_NCAL', '64'))     # calibration samples
CAL_CN0 = float(os.environ.get('RKNN_CAL_CN0', '60'))  # max C/N0 (dB-Hz)

def build_quant(name, A, B, Kc, fs, target):
    import tempfile, shutil
    from rknn.api import RKNN
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='rknn_q_')
    try:
        ds = gen_calib(A, B, Kc, fs, NCAL, os.path.join(tmp, 'cal'))
        shutil.copy(name + '.onnx', os.path.join(tmp, 'm.onnx'))
        os.chdir(tmp)
        r = RKNN(verbose=False)
        r.config(target_platform=target, optimization_level=3)
        hyb = [['Sr', 'P'], ['Si', 'P']]  # F_A^-1 and power in FP16
        if (r.load_onnx(model='m.onnx') != 0 or r.hybrid_quantization_step1(
            dataset=ds, custom_hybrid=hyb) != 0):
            print('RKNN build error: %s' % name)
            return
        r.release()
        r = RKNN(verbose=False)
        if (r.hybrid_quantization_step2(model_input='m.model',
            data_input='m.data', model_quantization_cfg='m.quantization.cfg') != 0 or
            r.export_rknn(name + '.rknn') != 0):
            print('RKNN build error: %s' % name)
        r.release()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

# main ------------------------------------------------------------------------
def main(argv):
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', 'linux', 'rknn')
    target, shapes, fs, i = 'rk3588', [], 16e6, 0
    while i < len(argv):
        if argv[i] == '-o' and i + 1 < len(argv):
            outdir = argv[i + 1]; i += 1
        elif argv[i] == '-t' and i + 1 < len(argv):
            target = argv[i + 1]; i += 1
        elif argv[i] == '-fs' and i + 1 < len(argv):
            fs = float(argv[i + 1]) * 1e6; i += 1
        else:
            shapes.append([int(v) for v in argv[i].split(':')])
        i += 1
    if not shapes:
        print('usage: gen_rknn_acq.py [-o dir] [-t target] [-fs fs] M:Kc[:R] ...')
        return 1
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    import onnx
    for s in shapes:
        M, Kc = s[0], s[1]
        A, B = factorize(M)
        if A == 0:
            print('unsupported DFT size: M=%d' % M)
            continue
        R = s[2] if len(s) > 2 else A // 2
        name = os.path.join(outdir, 'acq_M%d_K%d_R%d' % (M, Kc, R))
        onnx.save(build_onnx_shift(A, B, Kc, R), name + '.onnx')
        e = check_onnx_shift(name + '.onnx', A, B, Kc, R)
        print('M=%d (A=%d B=%d) Kc=%d R=%d: ONNX vs FFT NRMSE=%.2e' % (M, A,
            B, Kc, R, e))
        build_quant(name, A, B, Kc, fs, target)
        os.remove(name + '.onnx')
        print('-> %s.rknn' % name)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
