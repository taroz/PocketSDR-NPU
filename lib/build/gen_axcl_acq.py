#!/usr/bin/env python3
#
#  generate Axera axmodels of parallel code search for Axera NPU plugin
#  (libsdr_acc_axcl.so)
#
#  usage: gen_axcl_acq.py [-o dir] [-fs fs] [-w nw] [-npu mode] [-ncal n]
#             [-pulsar2 cmd] M:Kc[:R[:A]] ...
#
#    M  : DFT size (2 x samples per code period)
#    Kc : Doppler bins per inference (spaced by fs / M)
#    R  : output rows of IDFT (default: M / 2 / B, nout = R * B = M / 2 for the
#         search reduced to the peak)
#    A  : factorization M = A x B (default: factorize(M), A even preferred)
#    -o : output directory (default: ../linux/axcl)
#    -fs: sampling frequency (MHz) of the calibration signals (default: 8)
#    -w : windows per inference (default: 1, e.g. 4 for 1 ms codes: the
#         inference overhead of the device is about 0.2 ms)
#    -npu: NPU mode of AX650 (default: NPU3, 3 cores)
#    -ncal: calibration samples (default: 32)
#    -pulsar2: Pulsar2 command (default: pulsar2)
#
#  The model acq_M<M>_K<Kc>_R<R>_hs[_w<nw>]_red.axmodel (shared DFT model of
#  gen_hailo_acq.py, inputs xr, xi [1][nw*B][1][A] (int8, unitary DFT of the IF
#  samples of nw windows wiped by the carrier of the first bin, scaled to RMS
#  32), cr, ci [1][B][1][A] (int8, code DFT scaled to max 127), Pa (float,
#  accumulated power)) outputs P = Pa + sum of |corr|^2 / 64 of the windows
#  [1][(k,p1)][1][p2] (float) and the max, sum and index of max of P over p2
#  (Pmax, Psum, Parg [1][(k,p1)][1][1]), 16-bit quantized inside (U16, the
#  accumulation and the sums in FP32), calibrated by synthetic signals
#  (C/N0 30-60 dB-Hz) of the codes of the period M / 2 / fs.
#
#  Requires Pulsar2 (8.0, x86 Linux), onnx, onnxruntime and numpy.
#
#  Author:
#  T.SUZUKI
#
#  History:
#  2026-10-23  1.0  new
#
import sys, os, json, tarfile, subprocess, tempfile, shutil
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))
import gen_hailo_acq as gh
import sdr_code

SX, SC = 32.0, 127.0    # RMS of X inputs, max of code inputs (int8)
SIGS = ['L1CA', 'B1I', 'L5I', 'E5AI', 'E1B', 'L1CP', 'L2CM'] # calibration codes

# factorization M = A x B (A even for R = A / 2, nout = M / 2) ----------------
def factorize(M):
    for a in range(int(np.sqrt(M)), 7, -1):
        if M % a == 0 and a % 2 == 0: return a, M // a
    for a in range(int(np.sqrt(M)) + 1, M // 8 + 1):
        if M % a == 0 and a % 2 == 0: return a, M // a
    return 0, 0

# ONNX model --------------------------------------------------------------------
def build(A, B, Kc, R, nw):
    base = gh.build(A, B, Kc, R, True)  # P [1][(k,p1)][1][p2]
    g0 = base.graph
    inames = {t.name for t in g0.initializer}
    nodes, inits = [], list(g0.initializer)
    ins = [h.make_tensor_value_info('xr', T.FLOAT, [1, nw * B, 1, A]),
           h.make_tensor_value_info('xi', T.FLOAT, [1, nw * B, 1, A]),
           h.make_tensor_value_info('cr', T.FLOAT, [1, B, 1, A]),
           h.make_tensor_value_info('ci', T.FLOAT, [1, B, 1, A]),
           h.make_tensor_value_info('Pa', T.FLOAT, [1, Kc * R, 1, B])]
    def i64(n, v): inits.append(nh.from_array(np.array(v, dtype=np.int64), n))
    i64('ax1', [1]); i64('ax3', [3])
    outs = []
    for w in range(nw): # windows by Slice of the channels, the code shared
        def ren(t):
            if t in ('cr', 'ci') or t in inames or t == '': return t
            return 'w%d_%s' % (w, t)
        i64('s%d' % w, [w * B]); i64('e%d' % w, [(w + 1) * B])
        for k in ('xr', 'xi'):
            nodes.append(h.make_node('Slice', [k, 's%d' % w, 'e%d' % w, 'ax1'],
                ['w%d_%s' % (w, k)], name='slc_w%d_%s' % (w, k)))
        for n in g0.node:
            nn = onnx.NodeProto(); nn.CopyFrom(n)
            nn.name = 'w%d_%s' % (w, n.name)
            del nn.input[:]; nn.input.extend([ren(t) for t in n.input])
            del nn.output[:]; nn.output.extend([ren(t) for t in n.output])
            nodes.append(nn)
        outs.append('w%d_P' % w)
    s = outs[0]
    for w in range(1, nw):
        nodes.append(h.make_node('Add', [s, outs[w]], ['sum%d' % w], name='sum%d' % w))
        s = 'sum%d' % w
    nodes.append(h.make_node('Add', [s, 'Pa'], ['P'], name='acc_add'))
    nodes.append(h.make_node('ReduceMax', ['P'], ['Pmax'], name='red_max', axes=[3], keepdims=1))
    nodes.append(h.make_node('ReduceSum', ['P', 'ax3'], ['Psum'], name='red_sum', keepdims=1))
    nodes.append(h.make_node('ArgMax', ['P'], ['Parg'], name='red_arg', axis=3, keepdims=1))
    outv = [h.make_tensor_value_info('P', T.FLOAT, [1, Kc * R, 1, B])] + [
        h.make_tensor_value_info(n, t, [1, Kc * R, 1, 1]) for n, t in
        (('Pmax', T.FLOAT), ('Psum', T.FLOAT), ('Parg', T.INT64))]
    m = h.make_model(h.make_graph(nodes, 'sdr_acq_axcl', ins, outv, inits),
        opset_imports=[h.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m, base

# synthetic inputs of nw consecutive windows of a signal ------------------------
def sig_windows(rng, A, B, Kc, nw, fs, cn0, sig):
    M, N = A * B, A * B // 2; T_ = N / fs
    code = []
    while len(code) == 0: code = sdr_code.gen_code(sig, int(rng.integers(1, 33)))
    amp = np.sqrt(10 ** (cn0 / 10) / fs) if cn0 > 0 else 0.0
    fd = rng.uniform(-0.5, Kc - 0.5) * fs / M
    L = (nw + 1) * N; n = np.arange(L)
    rc = np.tile(sdr_code.res_code(code, T_, rng.uniform(0, T_), fs, N), nw + 1)
    x = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2) + \
        amp * rc * np.exp(2j * np.pi * fd * n / fs)
    c = sdr_code.gen_code_fft(code, T_, 0.0, fs, N, N); c /= np.max(np.abs(c))
    q8 = lambda v, s: np.clip(np.round(v * s), -128, 127).astype(np.float32)
    xs = []
    for w in range(nw):
        xw = x[w * N: w * N + M]; xw = xw / np.sqrt(np.mean(np.abs(xw) ** 2))
        xs.append(gh.xin(xw, A, B, True))
    cs = c.reshape(B, A)[None, :, None, :]
    return [np.concatenate([q8(v.real, SX) for v in xs], axis=1),
            np.concatenate([q8(v.imag, SX) for v in xs], axis=1), q8(cs.real, SC), q8(cs.imag, SC)]

def gen_inputs(rng, A, B, Kc, R, nw, fs, n, sbase, cmax=60.0):
    sigs = [s for s in SIGS if abs(sdr_code.code_cyc(s) - A * B / 2 / fs) < 1e-9]
    if not sigs: raise ValueError('no code for period %.3f ms' % (A * B / 2 / fs * 1e3))
    out = []
    for i in range(n):
        cn0 = (cmax if i % 8 == 1 else rng.uniform(30.0, cmax)) if i % 4 else 0.0 # 1/4 noise
                                                     # only, max C/N0 included
        q = sig_windows(rng, A, B, Kc, nw, fs, cn0, sigs[i % len(sigs)])
        p = sum(sbase.run(None, {'xr': q[0][:, w * B:(w + 1) * B], 'xi': q[1][:, w * B:(w + 1) * B],
            'cr': q[2], 'ci': q[3]})[0] for w in range(nw))
        q.append((p * rng.integers(0, 8)).astype(np.float32)) # accumulated 0-7 calls
        out.append(q)
    return out

# quantize and compile by Pulsar2 -----------------------------------------------
def gen_axmodel(onnx_file, base_file, name, A, B, Kc, R, nw, fs, out_dir, npu,
    ncal, pulsar2):
    import onnxruntime as ort
    sbase = ort.InferenceSession(base_file)
    ins = ['xr', 'xi', 'cr', 'ci', 'Pa']
    work = tempfile.mkdtemp(prefix='axcl_')
    try:
        cal = gen_inputs(np.random.default_rng(1), A, B, Kc, R, nw, fs, ncal, sbase)
        ic = []
        for j, k in enumerate(ins):
            t = os.path.join(work, 'cal_%s.tar' % k)
            with tarfile.open(t, 'w') as tf:
                for i, q in enumerate(cal):
                    f = os.path.join(work, 'tmp.npy'); np.save(f, q[j])
                    tf.add(f, arcname='%d.npy' % i)
            ic.append({'tensor_name': k, 'calibration_dataset': t, 'calibration_format': 'Numpy',
                'calibration_size': ncal, 'calibration_mean': [0], 'calibration_std': [1]})
        lc = [{'start_tensor_names': ['DEFAULT'], 'end_tensor_names': ['DEFAULT'], 'data_type': 'U16'},
              {'layer_names': ['acc_add', 'red_sum'] + ['sum%d' % w for w in range(1, nw)],
               'data_type': 'FP32'}]
        cfg = {'model_type': 'ONNX', 'npu_mode': npu, 'target_hardware': 'AX650',
            'quant': {'input_configs': ic, 'layer_configs': lc, 'calibration_method': 'MinMax',
                      'precision_analysis': False},
            'input_processors': [{'tensor_name': k, 'src_dtype': 'FP32' if k == 'Pa' else 'S8',
                                  'src_layout': 'NCHW'} for k in ins],
            'compiler': {'check': 0}}
        cf = os.path.join(work, 'config.json')
        json.dump(cfg, open(cf, 'w'), indent=1)
        r = subprocess.run([pulsar2, 'build', '--input', onnx_file, '--output_dir', work,
            '--config', cf], capture_output=True, text=True)
        log = r.stdout + r.stderr
        if r.returncode or not os.path.exists(os.path.join(work, 'compiled.axmodel')):
            print('%s: pulsar2 build error\n%s' % (name, log[-2000:])); return 0
        cyc = [l for l in log.split('\n') if 'max_cycle' in l]
        file = os.path.join(out_dir, name + '.axmodel')
        shutil.copy(os.path.join(work, 'compiled.axmodel'), file)
        print('%s: %d bytes, %s' % (file, os.path.getsize(file),
            cyc[-1].split('|')[-1].strip() if cyc else ''))
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

if __name__ == '__main__':
    out_dir, fs, nw, npu, ncal, pulsar2, shapes = None, 8e6, 1, 'NPU3', 32, 'pulsar2', []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o': out_dir = sys.argv[i + 1]; i += 1
        elif sys.argv[i] == '-fs': fs = float(sys.argv[i + 1]) * 1e6; i += 1
        elif sys.argv[i] == '-w': nw = int(sys.argv[i + 1]); i += 1
        elif sys.argv[i] == '-npu': npu = sys.argv[i + 1]; i += 1
        elif sys.argv[i] == '-ncal': ncal = int(sys.argv[i + 1]); i += 1
        elif sys.argv[i] == '-pulsar2': pulsar2 = sys.argv[i + 1]; i += 1
        else: shapes.append([int(v) for v in sys.argv[i].split(':')])
        i += 1
    if not shapes:
        print('usage: gen_axcl_acq.py [-o dir] [-fs fs] [-w nw] [-npu mode] [-ncal n] '
              '[-pulsar2 cmd] M:Kc[:R[:A]] ...')
        sys.exit(1)
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../linux/axcl')
    os.makedirs(out_dir, exist_ok=True)
    for shape in shapes:
        M, Kc = shape[:2]
        A, B = (shape[3], M // shape[3]) if len(shape) > 3 else factorize(M)
        if A == 0 or A * B != M:
            print('invalid factorization of M=%d' % M); continue
        R = shape[2] if len(shape) > 2 else (M // 2 + B - 1) // B
        name = 'acq_M%d_K%d_R%d_hs%s_red' % (M, Kc, R, '_w%d' % nw if nw > 1 else '')
        m, base = build(A, B, Kc, R, nw)
        file = os.path.join(out_dir, name + '.onnx')
        bfile = os.path.join(out_dir, name + '_base.onnx')
        onnx.save(m, file); onnx.save(base, bfile)
        print('%s: A=%d B=%d base ONNX vs FFT NRMSE=%.2e' % (name, A, B,
            gh.check(bfile, A, B, Kc, R, True)))
        gen_axmodel(file, bfile, name, A, B, Kc, R, nw, fs, out_dir, npu, ncal, pulsar2)
        os.remove(file); os.remove(bfile)
