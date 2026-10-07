#!/usr/bin/env python3
#
#  generate RKNN models of parallel code search for Rockchip NPU plugin
#  (libsdr_acc_rknn.so)
#
#  usage: gen_rknn_acq.py [-o dir] [-t target] [-w|-f|-q] [-fs fs] M:Kc[:R] ...
#
#    M  : DFT size (2 x samples per code period)
#    Kc : Doppler bins per inference (= ceil(K / NPU cores), K: bins of search)
#    R  : output rows of IDFT (default: A / 2, A x B = M, nout = R * B)
#    -o : output directory (default: ../linux/rknn)
#    -t : target platform (default: rk3588)
#    -w : carrier wipe-off in the model (inputs: IF samples xr, xi [1][n1][1][n2]
#         shared by bins and separable carrier ar, ai [Kc][n1][1][1], br, bi
#         [Kc][1][1][n2]) (model name: acq_M<M>_K<Kc>_R<R>_w.rknn)
#    -f : carrier folded into the DFT (inputs: IF samples xr, xi [1][n1][1][n2]
#         wiped by the carrier of the first bin, code cr, ci [1][k2][1][W]
#         replicated along W, Doppler bins spaced by fs / M) (model name:
#         acq_M<M>_K<Kc>_R<R>_f.rknn)
#
#    -q : INT8 model of -f (acq_M<M>_K<Kc>_R<R>_fq.rknn): all ops INT8 except
#         the last IDFT stage F_A^-1 and the correlation power (FP16) (hybrid
#         quantization), calibrated by synthetic signals (C/N0 30-60 dB-Hz) of the
#         codes of the period M / 2 / fs (-fs: sampling frequency in MHz,
#         default 16), code DFT normalized by max magnitude
#
#  Carrier folding (-f): with the Doppler bins spaced by the DFT bin fs / M,
#  the carrier of bin k is the carrier of the first bin times W_M^(k*n), and
#  for n = B*n1+n2, W_M^(k*n) = W_A^(k*n1) * W_M^(k*n2). The first factor is
#  folded into the F_A weights of bin k (rows rotated by k, bins stacked along
#  the output channels of one Conv) and the second into the twiddle constants
#  of bin k. Each bin still has its own full DFT on the NPU (no shared DFT,
#  same MACs as the direct algorithm), but the element-wise carrier wipe-off is
#  removed. If Kc * A <= MAX_W, the bins are kept in the channels/width with
#  batch 1 ([1][(k,k1)][1][n2] -> Transpose -> [1][n2][1][(k,k1)]), which makes
#  the Transposes ~2x faster than with the bins in the batch. The complex Conv
#  is computed by 3 real Conv (Wr(Xr+Xi), (Wr+Wi)Xi, (Wi-Wr)Xr).#
#  e.g. gen_rknn_acq.py 32000:7 128000:27  (16 MHz, 1 ms and 4 ms codes,
#                                           21 and 81 Doppler bins, 3 cores)
#
#  Requires RKNN-Toolkit2 (pip install rknn-toolkit2, same version as the RKNN
#  runtime of the target), onnx and numpy.
#
#  The graph is the four-step DFT of the code search (direct mode: carrier
#  wipe-off and DFT for each Doppler bin, same algorithm as the CPU reference)
#  in the NCHW layout of the NPU, so that each DFT stage is a 1x1 Conv over the
#  channels (complex: 4 real Conv) and the axis changes are 2 Transposes:
#
#    xr, xi [Kc][n1][1][n2] (carrier wiped, x[B*n1+n2], FP16, unit RMS)
#      -> F_A (Conv) -> twiddle W_M^(k1*n2) -> Transpose -> F_B (Conv)
#      -> * code (cr, ci [1][k2][1][k1] = code[k1+A*k2])
#      -> F_B^-1 (Conv) -> twiddle W_M^-(k1*p2) -> Transpose -> F_A^-1 (Conv)
#      -> P [Kc][p1][1][p2] = |c[B*p1+p2]|^2 / 64 (p1 < R)
#
#  The DFTs are unitary. The output powers are scaled by 1/64 to keep them in
#  the FP16 range.
#
import sys, os
import numpy as np

MIN_FACT = 8
MAX_W = 8192    # max width of NPU tensors (larger: CPU fallback)

# factorize DFT size (same as sdr_acc_rknn.cpp) -------------------------------
def factorize(M):
    for a in range(int(np.sqrt(M)), MIN_FACT - 1, -1):
        if M % a == 0:
            return a, M // a
    return 0, 0

# ONNX graph ------------------------------------------------------------------
def build_onnx(A, B, Kc, R, wipe=False):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    M = A * B
    nodes, inits, cnt = [], [], [0]

    def nm(p):
        cnt[0] += 1
        return '%s_%d' % (p, cnt[0])

    def const(v):
        name = nm('c')
        inits.append(nh.from_array(np.asarray(v, dtype=np.float32), name))
        return name

    def op(t, ins, **kw):
        o = nm(t)
        nodes.append(h.make_node(t, ins, [o], **kw))
        return o

    def dft(n, sgn):  # w[k][n] (out, in) unitary
        k = np.arange(n)[:, None] * np.arange(n)[None, :]
        return np.exp(sgn * 2j * np.pi * (k % n) / n) / np.sqrt(n)

    def cconv(xr, xi, w):  # complex 1x1 Conv over channels
        wr = const(w.real[:, :, None, None])
        wi = const(w.imag[:, :, None, None])
        wn = const(-w.imag[:, :, None, None])
        yr = op('Add', [op('Conv', [xr, wr]), op('Conv', [xi, wn])])
        yi = op('Add', [op('Conv', [xr, wi]), op('Conv', [xi, wr])])
        return yr, yi

    def cmul(ar, ai, br, bi):
        r = op('Sub', [op('Mul', [ar, br]), op('Mul', [ai, bi])])
        i = op('Add', [op('Mul', [ar, bi]), op('Mul', [ai, br])])
        return r, i

    shapes = [('xr', [Kc, A, 1, B]), ('xi', [Kc, A, 1, B]), ('cr', [1, B, 1, A]),
         ('ci', [1, B, 1, A])]
    if wipe:
        shapes = [('xr', [1, A, 1, B]), ('xi', [1, A, 1, B]),
            ('ar', [Kc, A, 1, 1]), ('ai', [Kc, A, 1, 1]), ('br', [Kc, 1, 1, B]),
            ('bi', [Kc, 1, 1, B])] + shapes[2:]
    ins = [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in shapes]
    k1 = np.arange(A)[:, None]
    n2 = np.arange(B)[None, :]
    t1 = np.exp(-2j * np.pi * (k1 * n2 % M) / M)    # [k1][n2]
    t2 = np.exp(2j * np.pi * (n2.T * k1.T % M) / M) # [p2][k1]

    xr, xi = 'xr', 'xi'
    if wipe: # carrier x * a[n1] * b[n2]
        xr, xi = cmul(xr, xi, 'ar', 'ai')           # [Kc][n1][1][n2]
        xr, xi = cmul(xr, xi, 'br', 'bi')
    yr, yi = cconv(xr, xi, dft(A, -1))              # [Kc][k1][1][n2]
    zr, zi = cmul(yr, yi, const(t1.real[None, :, None, :]),
        const(t1.imag[None, :, None, :]))
    zr = op('Transpose', [zr], perm=[0, 3, 2, 1])   # [Kc][n2][1][k1]
    zi = op('Transpose', [zi], perm=[0, 3, 2, 1])
    ur, ui = cconv(zr, zi, dft(B, -1))              # [Kc][k2][1][k1]
    vr, vi = cmul(ur, ui, 'cr', 'ci')
    qr, qi = cconv(vr, vi, dft(B, 1))               # [Kc][p2][1][k1]
    sr, si = cmul(qr, qi, const(t2.real[None, :, None, :]),
        const(t2.imag[None, :, None, :]))
    sr = op('Transpose', [sr], perm=[0, 3, 2, 1])   # [Kc][k1][1][p2]
    si = op('Transpose', [si], perm=[0, 3, 2, 1])
    orr, oi = cconv(sr, si, dft(A, 1)[:R] / 8.0)    # [Kc][p1][1][p2]
    op('Add', [op('Mul', [orr, orr]), op('Mul', [oi, oi])])
    nodes[-1].output[0] = 'P'
    outs = [h.make_tensor_value_info('P', T.FLOAT, [Kc, R, 1, B])]
    g = h.make_graph(nodes, 'sdr_acq', ins, outs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

# ONNX graph with carrier folding (-f) ------------------------------------------
def build_onnx_fold(A, B, Kc, R):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    M, W = A * B, Kc * A
    flat = W <= MAX_W   # bins in channels/width (batch 1)
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
    
    Wc = W if flat else A
    ins = [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in
        [('xr', [1, A, 1, B]), ('xi', [1, A, 1, B]), ('cr', [1, B, 1, Wc]),
         ('ci', [1, B, 1, Wc])]]
    k1 = np.arange(A)[None, :, None]
    n2 = np.arange(B)[None, None, :]
    kk = np.arange(Kc)[:, None, None]
    G = np.concatenate([dft(A, -1, (np.arange(A) + k) % A) for k in range(Kc)])
    t1 = np.exp(-2j * np.pi * (((k1 + kk) * n2) % M) / M)  # [k][k1][n2]
    t2 = np.exp(2j * np.pi * ((n2[0].T * k1[0].T) % M) / M) # [p2][k1]
    
    yr, yi = cconv('xr', 'xi', G)                       # [1][(k,k1)][1][n2]
    if flat:
        t1 = t1.reshape(1, W, 1, B)
        yr, yi = cmul(yr, yi, const(t1.real), const(t1.imag))
        perm = [0, 3, 2, 1]
        zr, zi = op('Transpose', [yr], perm=perm), op('Transpose', [yi], perm=perm)
        ur, ui = cconv(zr, zi, dft(B, -1, range(B)))    # [1][k2][1][(k,k1)]
        vr, vi = cmul(ur, ui, 'cr', 'ci')
        qr, qi = cconv(vr, vi, dft(B, 1, range(B)))     # [1][p2][1][(k,k1)]
        t2 = np.tile(t2, (1, Kc))[None, :, None, :]
        sr, si = cmul(qr, qi, const(t2.real), const(t2.imag))
        sr, si = op('Transpose', [sr], perm=perm), op('Transpose', [si], perm=perm)
        sr, si = reshape(sr, [Kc, A, 1, B]), reshape(si, [Kc, A, 1, B])
    else:
        yr, yi = reshape(yr, [Kc, A, 1, B]), reshape(yi, [Kc, A, 1, B])
        t1 = t1[:, :, None, :]
        yr, yi = cmul(yr, yi, const(t1.real), const(t1.imag))
        perm = [0, 3, 2, 1]
        zr, zi = op('Transpose', [yr], perm=perm), op('Transpose', [yi], perm=perm)
        ur, ui = cconv(zr, zi, dft(B, -1, range(B)))    # [Kc][k2][1][k1]
        vr, vi = cmul(ur, ui, 'cr', 'ci')
        qr, qi = cconv(vr, vi, dft(B, 1, range(B)))     # [Kc][p2][1][k1]
        t2 = t2[None, :, None, :]
        sr, si = cmul(qr, qi, const(t2.real), const(t2.imag))
        sr, si = op('Transpose', [sr], perm=perm), op('Transpose', [si], perm=perm)
    nodes[-2].output[0], nodes[-1].output[0] = 'Sr', 'Si' # (for hybrid quant.)
    sr, si = 'Sr', 'Si'
    orr, oi = cconv(sr, si, dft(A, 1, range(R)) / 8.0)  # [Kc][p1][1][p2]
    nodes[-3].output[0] = 'Or'  # (named for hybrid quantization)
    nodes[-1].output[0] = 'Oi'
    op('Add', [op('Mul', ['Or', 'Or']), op('Mul', ['Oi', 'Oi'])])
    nodes[-1].output[0] = 'P'
    outs = [h.make_tensor_value_info('P', T.FLOAT, [Kc, R, 1, B])]
    g = h.make_graph(nodes, 'sdr_acq_f', ins, outs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

# check ONNX graph of carrier folding by FFT (bins spaced by fs / M) ----------
def check_onnx_fold(file, A, B, Kc, R):
    import onnxruntime as ort
    M, W = A * B, Kc * A
    rng = np.random.default_rng(1)
    x = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    f0, n = 0.3141, np.arange(M)  # carrier of first bin (cyc/sample)
    xw = (x * np.exp(-2j * np.pi * f0 * n)).reshape(1, A, 1, B)
    cs = c.reshape(B, A)
    cs = (np.tile(cs, (1, Kc)) if W <= MAX_W else cs)[None, :, None, :]
    o = ort.InferenceSession(file).run(None, {k: v.astype(np.float32) for k, v in
        {'xr': xw.real, 'xi': xw.imag, 'cr': cs.real, 'ci': cs.imag}.items()})[0]
    ref = np.array([np.abs(np.fft.ifft(np.fft.fft(x * np.exp(-2j * np.pi *
        (f0 + k / M) * n)) * c)) ** 2 / 64 for k in range(Kc)])[:, :R * B]
    got = o.reshape(Kc, R * B)
    return np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref ** 2))

# calibration dataset for INT8 model (-q) ---------------------------------------
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
    M, N, W = A * B, A * B // 2, Kc * A if Kc * A <= MAX_W else A
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
        cs = c.reshape(B, A)
        cs = (np.tile(cs, (1, W // A)))[None, :, None, :]
        fs_ = []
        for k, v in [('xr', xs.real), ('xi', xs.imag), ('cr', cs.real), ('ci', cs.imag)]:
            f = os.path.join(dir, '%s_%d.npy' % (k, i))
            np.save(f, v.astype(np.float32))
            fs_.append(f)
        lines.append(' '.join(fs_))
    file = os.path.join(dir, 'dataset.txt')
    open(file, 'w').write('\n'.join(lines) + '\n')
    return file

# check ONNX graph by FFT -----------------------------------------------------
def check_onnx(file, A, B, Kc, R, wipe=False):
    import onnxruntime as ort
    M = A * B
    rng = np.random.default_rng(1)
    x = rng.standard_normal((Kc, M)) + 1j * rng.standard_normal((Kc, M))
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    xs = x.reshape(Kc, A, 1, B)
    cs = c.reshape(B, 1, A)[None]
    feed = {'xr': xs.real, 'xi': xs.imag, 'cr': cs.real, 'ci': cs.imag}
    if wipe: # x = x0 * a * b
        x0 = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        a = np.exp(2j * np.pi * rng.random((Kc, A, 1, 1)))
        b = np.exp(2j * np.pi * rng.random((Kc, 1, 1, B)))
        x = (x0.reshape(1, A, 1, B) * a * b).reshape(Kc, M)
        feed.update({'xr': x0.real.reshape(1, A, 1, B),
            'xi': x0.imag.reshape(1, A, 1, B), 'ar': a.real, 'ai': a.imag,
            'br': b.real, 'bi': b.imag})
    o = ort.InferenceSession(file).run(None, {k: v.astype(np.float32)
        for k, v in feed.items()})[0]
    ref = np.abs(np.fft.ifft(np.fft.fft(x, axis=1) * c[None], axis=1)) ** 2 / 64
    ref = ref[:, :R * B]
    got = o.reshape(Kc, R * B)
    return np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref ** 2))

# build INT8 model (hybrid: correlation power in FP16) ------------------------
NCAL = int(os.environ.get('RKNN_NCAL', '64'))     # calibration samples
CAL_CN0 = float(os.environ.get('RKNN_CAL_CN0', '60'))  # max C/N0 (dB-Hz)
QFLOAT = os.environ.get('RKNN_QFLOAT', 'F')   # FP16 part: F (IDFT F_A^-1 + power), P (power)

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
        hyb = [['Or', 'P'], ['Oi', 'P']] if QFLOAT == 'P' else [['Sr', 'P'], ['Si', 'P']]
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
    target, shapes, wipe, fold, quant, fs, i = 'rk3588', [], False, False, False, 16e6, 0
    while i < len(argv):
        if argv[i] == '-o' and i + 1 < len(argv):
            outdir = argv[i + 1]; i += 1
        elif argv[i] == '-t' and i + 1 < len(argv):
            target = argv[i + 1]; i += 1
        elif argv[i] == '-w':
            wipe = True
        elif argv[i] == '-f':
            fold = True
        elif argv[i] == '-q':
            fold = quant = True
        elif argv[i] == '-fs' and i + 1 < len(argv):
            fs = float(argv[i + 1]) * 1e6; i += 1
        else:
            shapes.append([int(v) for v in argv[i].split(':')])
        i += 1
    if not shapes:
        print('usage: gen_rknn_acq.py [-o dir] [-t target] [-w|-f|-q] [-fs fs] M:Kc[:R] ...')
        return 1
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    import onnx
    from rknn.api import RKNN
    for s in shapes:
        M, Kc = s[0], s[1]
        A, B = factorize(M)
        if A == 0:
            print('unsupported DFT size: M=%d' % M)
            continue
        R = s[2] if len(s) > 2 else A // 2
        name = os.path.join(outdir, 'acq_M%d_K%d_R%d%s' % (M, Kc, R,
            '_fq' if quant else ('_f' if fold else ('_w' if wipe else ''))))
        if fold:
            onnx.save(build_onnx_fold(A, B, Kc, R), name + '.onnx')
            e = check_onnx_fold(name + '.onnx', A, B, Kc, R)
        else:
            onnx.save(build_onnx(A, B, Kc, R, wipe), name + '.onnx')
            e = check_onnx(name + '.onnx', A, B, Kc, R, wipe)
        print('M=%d (A=%d B=%d) Kc=%d R=%d: ONNX vs FFT NRMSE=%.2e' % (M, A,
            B, Kc, R, e))
        if quant:
            build_quant(name, A, B, Kc, fs, target)
        else:
            r = RKNN(verbose=False)
            r.config(target_platform=target, optimization_level=3)
            if (r.load_onnx(model=name + '.onnx') != 0 or
                r.build(do_quantization=False) != 0 or
                r.export_rknn(name + '.rknn') != 0):
                print('RKNN build error: %s' % name)
            r.release()
        os.remove(name + '.onnx')
        print('-> %s.rknn' % name)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
