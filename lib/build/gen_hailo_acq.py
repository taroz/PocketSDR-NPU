#!/usr/bin/env python3
#
#  generate Hailo HEF models of parallel code search for Hailo NPU plugin
#  (libsdr_acc_hailo.so)
#
#  usage: gen_hailo_acq.py [-arch arch] [-o dir] [-fs fs] [-sig sig,...] [-n]
#             [-q8] [-ms cmds] M:Kc:R[:A] ...
#
#    M  : DFT size (2 x samples per code period)
#    Kc : Doppler bins per inference (spaced by fs / M)
#    R  : output rows of IDFT (nout = R * B <= M / 2, A x B = M), padded to
#         avoid the corrupted outputs of some multi-context HEFs (verified:
#         48000:7:112 for 24 MHz, 16000:7:64, 64000:5:168:320, 160000:1:200,
#         320000:1:200:400 for 8 MHz)
#    A  : factorization M = A x B (default: factorize(M)), A, B multiples of 8
#         for the inputs in the native order of the device (e.g. 320000:1:200:400,
#         the plugin tries A, B multiples of 8 first, then factorize(M))
#    -arch: target device (hailo10h: Dataflow Compiler 5.x, hailo8: 3.x)
#         (default: hailo10h)
#    -o : output directory (default: ../linux/hailo, ../linux/hailo8 for hailo8)
#    -fs: sampling frequency (MHz) of the calibration signals (default: 24)
#    -sig: codes of the calibration signals, the first two also evaluated
#         (default: L1CA,L5I,B1I,E5AI, e.g. E1B,E1C for 64000 at 8 MHz)
#    -n : no compile (ONNX, quantization and emulator evaluation only)
#    -q8: all layers 8-bit (no 16-bit tail)
#    -ms: model script commands added, separated by ';'
#
#  The model (inputs xr, xi [1][n1][1][n2], IF samples wiped by the carrier of
#  the first bin and normalized by RMS, cr, ci [1][k2][1][k1], code DFT
#  normalized by max magnitude) computes the carrier folded four-step DFT of
#  the Kc bins (bins in the channels), code multiplication, IDFT and the
#  correlation powers of two copies of the last IDFT outputs (16-bit tail)
#  (output P [1][(k,p1)][1][p2] = |corr|^2 / 64). The INT8 model is calibrated
#  by synthetic signals of the 1 ms codes (L1CA, L5I, B1I, E5AI) of C/N0 30-58
#  dB-Hz (the codes of 10.23 Mcps are required to avoid the clipped peaks).
#
#  The Hailo Dataflow Compiler (hailo_sdk_client, x86 Linux, Python 3.10) and
#  onnx, onnxruntime are required. Verify the HEF on the device by the peak
#  positions of all code offsets (see src/sdr_acc_hailo.cpp).
#
#  Author:
#  T.SUZUKI
#
#  History:
#  2026-10-11  1.0  new
#  2026-10-11  1.1  add option -sig
#  2026-10-13  1.2  add factorization A in M:Kc:R:A
#  2026-10-14  1.3  add options -arch, -q8, -ms
#
import sys, os
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))
from gen_rknn_acq import factorize

SIGS = ['L1CA', 'L5I', 'B1I', 'E5AI'] # codes of calibration signals

# build ONNX model --------------------------------------------------------------
def build(A, B, Kc, R):
    M, W = A * B, Kc * A
    nodes, inits, cnt = [], [], [0]
    def nm(p):
        cnt[0] += 1; return '%s_%d' % (p, cnt[0])
    def const(v):
        name = nm('c'); inits.append(nh.from_array(np.asarray(v, dtype=np.float32), name)); return name
    def op(t, ins, out=None, **kw):
        o = out or nm(t); nodes.append(h.make_node(t, ins, [o], name=nm('n_' + t), **kw)); return o
    def dft(n, sgn, rows):
        k = np.asarray(rows)[:, None] * np.arange(n)[None, :]
        return np.exp(sgn * 2j * np.pi * (k % n) / n) / np.sqrt(n)
    def cconv(xr, xi, w, group=1): # complex 1x1 conv by 3 real convs
        c1 = op('Conv', [op('Add', [xr, xi]), const(w.real[:, :, None, None])], group=group, kernel_shape=[1, 1])
        yr = op('Add', [op('Conv', [xi, const(-(w.real + w.imag)[:, :, None, None])], group=group, kernel_shape=[1, 1]), c1])
        yi = op('Add', [op('Conv', [xr, const((w.imag - w.real)[:, :, None, None])], group=group, kernel_shape=[1, 1]), c1])
        return yr, yi
    def cmul(ar, ai, br, bi):
        return (op('Sub', [op('Mul', [ar, br]), op('Mul', [ai, bi])]),
                op('Add', [op('Mul', [ar, bi]), op('Mul', [ai, br])]))
    ins = [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in
        [('xr', [1, A, 1, B]), ('xi', [1, A, 1, B]), ('cr', [1, B, 1, A]), ('ci', [1, B, 1, A])]]
    k1 = np.arange(A)[None, :, None]; n2 = np.arange(B)[None, None, :]; kk = np.arange(Kc)[:, None, None]
    G = np.concatenate([dft(A, -1, (np.arange(A) + k) % A) for k in range(Kc)])
    t1 = np.exp(-2j * np.pi * (((k1 + kk) * n2) % M) / M).reshape(1, W, 1, B)
    t2 = np.exp(2j * np.pi * ((n2[0].T * k1[0].T) % M) / M)
    yr, yi = cconv('xr', 'xi', G)                                     # [1][(k,k1)][1][n2]
    yr, yi = cmul(yr, yi, const(t1.real), const(t1.imag))
    perm = [0, 3, 2, 1]
    zr, zi = op('Transpose', [yr], perm=perm), op('Transpose', [yi], perm=perm)
    ur, ui = cconv(zr, zi, dft(B, -1, range(B)))                      # [1][k2][1][(k,k1)]
    cr = op('Concat', ['cr'] * Kc, axis=3) if Kc > 1 else 'cr'
    ci = op('Concat', ['ci'] * Kc, axis=3) if Kc > 1 else 'ci'
    vr, vi = cmul(ur, ui, cr, ci)
    qr, qi = cconv(vr, vi, dft(B, 1, range(B)))                       # [1][p2][1][(k,k1)]
    t2 = np.tile(t2, (1, Kc))[None, :, None, :]
    sr, si = cmul(qr, qi, const(t2.real), const(t2.imag))
    sr, si = op('Transpose', [sr], perm=perm), op('Transpose', [si], perm=perm)  # [1][(k,k1)][1][p2]
    wl = np.concatenate([dft(A, 1, range(R)) / 8.0] * Kc)
    o1r, o1i = cconv(sr, si, wl, group=Kc)                            # two copies of the outputs
    o2r, o2i = cconv(sr, si, wl, group=Kc)
    op('Add', [op('Mul', [o1r, o2r]), op('Mul', [o1i, o2i])], 'P')    # [1][(k,p1)][1][p2]
    outs = [h.make_tensor_value_info('P', T.FLOAT, [1, Kc * R, 1, B])]
    m = h.make_model(h.make_graph(nodes, 'sdr_acq_h', ins, outs, inits),
        opset_imports=[h.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

# check ONNX model by FFT -------------------------------------------------------
def check(file, A, B, Kc, R):
    import onnxruntime as ort
    M = A * B; rng = np.random.default_rng(1); n = np.arange(M)
    x = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * np.exp(-2j * np.pi * 0.3141 * n)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    xs, cs = x.reshape(A, B)[None, :, None, :], c.reshape(B, A)[None, :, None, :]
    P = ort.InferenceSession(file).run(None, {k: v.astype(np.float32) for k, v in
        {'xr': xs.real, 'xi': xs.imag, 'cr': cs.real, 'ci': cs.imag}.items()})[0]
    P = P.astype(np.float64).reshape(Kc, R * B)
    ref = np.array([(np.abs(np.fft.ifft(np.fft.fft(x * np.exp(-2j * np.pi * k / M * n)) * c)) ** 2
        / 64)[:R * B] for k in range(Kc)])
    return np.sqrt(np.mean((P - ref) ** 2) / np.mean(ref ** 2))

# synthetic signal inputs of a window -------------------------------------------
def sig_input(rng, A, B, Kc, fs, cn0, sig, fd_bins=None):
    import sdr_code
    M, N = A * B, A * B // 2; T = N / fs
    code = sdr_code.gen_code(sig, int(rng.integers(1, 33)))
    amp = np.sqrt(10 ** (cn0 / 10) / fs) if cn0 > 0 else 0.0
    fd = (rng.uniform(-0.5, Kc - 0.5) if fd_bins is None else fd_bins) * fs / M
    rc = np.tile(sdr_code.res_code(code, T, rng.uniform(0, T), fs, N), 2)
    x = (rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)) / np.sqrt(2) + \
        amp * rc * np.exp(2j * np.pi * fd * np.arange(2 * N) / fs)
    x = x[:M] / np.sqrt(np.mean(np.abs(x[:M]) ** 2))
    c = sdr_code.gen_code_fft(code, T, 0.0, fs, N, N); c /= np.max(np.abs(c))
    xs, cs = x.reshape(A, B)[None, :, None, :], c.reshape(B, A)[None, :, None, :]
    return [xs.real, xs.imag, cs.real, cs.imag], x, c

# quantize, evaluate and compile by Dataflow Compiler ---------------------------
def gen_hef(onnx_file, name, A, B, Kc, R, fs, out_dir, nocomp, arch='hailo10h',
    q8=False, ms=''):
    from hailo_sdk_client import ClientRunner, InferenceContext
    r = ClientRunner(hw_arch=arch)
    r.translate_onnx_model(onnx_file, name)
    hn = r.get_hn_dict()['layers']
    ins = sorted(k for k, v in hn.items() if v['type'] == 'input_layer') # xr, xi, cr, ci
    outl = [k for k, v in hn.items() if v['type'] == 'output_layer'][0]
    tail, frontier = [outl], [outl] # output <- add <- 2 mults <- adds (16-bit)
    for _ in range(3):
        frontier = [i for l in frontier for i in hn[l]['input']]
        tail += frontier
    tail = sorted(set(l.split('/')[-1] for l in tail))
    nhwc = lambda a: np.transpose(a, (0, 2, 3, 1)).astype(np.float32)
    rng = np.random.default_rng(1)
    cal = [[] for _ in range(4)]
    for i in range(64):
        d, _, _ = sig_input(rng, A, B, Kc, fs, rng.uniform(30, 58) if i % 4 else 0,
            SIGS[(i // 4) % len(SIGS)])
        for j in range(4): cal[j].append(nhwc(d[j])[0])
    r.load_model_script('model_optimization_flavor(optimization_level=0, compression_level=0)\n' +
        ('' if q8 else 'quantization_param([%s], precision_mode=a16_w16)\n' % ', '.join(tail)) +
        ''.join(l + '\n' for l in ms.split(';') if l))
    r.optimize({ins[j]: np.stack(cal[j]) for j in range(4)})
    print('sig  C/N0  peak_pos  dCN0(q-ref) mean/std  NRMSE  peak(q/ref)')
    M, N = A * B, A * B // 2
    with r.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for sig in SIGS[:2]:
            for cn0 in (36, 40, 45, 50, 55):
                pk, dc, er, pr = 0, [], [], []
                for _ in range(5):
                    d, x, c = sig_input(rng, A, B, Kc, fs, cn0, sig, fd_bins=0.0)
                    ref = (np.abs(np.fft.ifft(np.fft.fft(x) * c)) ** 2 / 64)[:R * B]
                    q = np.asarray(r.infer(ctx, {ins[j]: nhwc(d[j]) for j in range(4)}))
                    q = np.transpose(q[0], (2, 0, 1)).reshape(Kc, -1)[0].astype(np.float64)
                    ir, iq = int(np.argmax(ref)), int(np.argmax(q))
                    cr = 10 * np.log10((ref[ir] - ref.mean()) / ref.mean() / (N / fs))
                    cq = 10 * np.log10(max(q[iq] - q.mean(), 1e-12) / q.mean() / (N / fs))
                    pk += abs(ir - iq) <= 1; dc.append(cq - cr); pr.append(q[ir] / ref[ir])
                    er.append(np.sqrt(np.mean((q - ref) ** 2) / np.mean(ref ** 2)))
                print('%-4s %4d  %d/5  %+.2f / %.2f dB  %.3f  %.2f' % (sig, cn0, pk, np.mean(dc),
                    np.std(dc), np.mean(er), np.mean(pr)))
    if not nocomp:
        hef = r.compile()
        file = os.path.join(out_dir, name + '.hef')
        open(file, 'wb').write(hef)
        print('%s: %d bytes' % (file, len(hef)))

if __name__ == '__main__':
    out_dir, arch, q8, ms = None, 'hailo10h', False, ''
    fs, nocomp, shapes = 24e6, False, []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o': out_dir = sys.argv[i + 1]; i += 1
        elif sys.argv[i] == '-arch': arch = sys.argv[i + 1]; i += 1
        elif sys.argv[i] == '-fs': fs = float(sys.argv[i + 1]) * 1e6; i += 1
        elif sys.argv[i] == '-sig': SIGS[:] = sys.argv[i + 1].split(','); i += 1
        elif sys.argv[i] == '-n': nocomp = True
        elif sys.argv[i] == '-q8': q8 = True
        elif sys.argv[i] == '-ms': ms = sys.argv[i + 1]; i += 1
        else: shapes.append([int(v) for v in sys.argv[i].split(':')])
        i += 1
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../linux/hailo8' if arch == 'hailo8' else '../linux/hailo')
    os.makedirs(out_dir, exist_ok=True)
    for shape in shapes:
        M, Kc, R = shape[:3]
        A, B = (shape[3], M // shape[3]) if len(shape) > 3 else factorize(M)
        if A * B != M:
            print('invalid factorization %d = %d x %d' % (M, A, B)); continue
        name = 'acq_M%d_K%d_R%d_h' % (M, Kc, R)
        file = os.path.join(out_dir, name + '.onnx')
        onnx.save(build(A, B, Kc, R), file)
        print('%s: A=%d B=%d ONNX vs FFT NRMSE=%.2e' % (name, A, B, check(file, A, B, Kc, R)))
        gen_hef(file, name, A, B, Kc, R, fs, out_dir, nocomp, arch, q8, ms)
