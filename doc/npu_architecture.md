# NPU Acceleration Architecture (Development Note)

Date: 2026-10-02 (section 13: 2026-10-07)

----

This note describes the architecture of the accelerator (NPU) backends of this
research fork of PocketSDR and the results of Phase 1 (NPU acquisition).

## 1. Motivation

The research question is:

```text
Can a commodity NPU enable real-time, all-in-view, multi-frequency GNSS
software-defined reception on a low-power SBC?
```

An all-in-view, multi-frequency, multi-GNSS receiver needs hundreds of signal
channels (frequencies x data/pilot components x constellations). The NPU is
used as a massively parallel signal processing accelerator (not for neural
network inference) for acquisition and tracking. The intended architectural
contribution is:

```text
conventional channel-oriented GNSS SDR
                ↓
batched / tensor-oriented GNSS signal processing
                ↓
massively parallel NPU execution
```

## 2. Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | NPU acquisition as an alternative backend of the existing code search, signal/channel-wise (no PRN batch) | implemented (this note) |
| 2 | Batched tracking correlator: `track_prepare` / batched correlator (CPU or NPU) / `track_finish`, batches grouped by RF frontend, sampling rate and primary code period | implemented (section 11) |
| 3 | Unified 1 ms micro-epoch NPU tracking (1 ms partial correlations + signal-specific coherent accumulation) | not started |

Phase 2 starts only after Phase 1 is verified; Phase 3 only after Phase 2.

## 3. Backend Architecture

```text
core PocketSDR GNSS logic (unchanged)
    search_sig() (sdr_ch.c)
        |
        v
    sdr_search_code() / sdr_search_code2()           sdr_func.c (dispatcher)
        |
        v
    sdr_acc_search_code()                            sdr_acc.c (generic layer)
        |
  +-----+--------------+-------------------------+
  |                    |                         |
 cpu (reference)     cpu_opt                  plugin (sdr_acc.h)
 sdr_search_code_cpu()  shared DFT             libsdr_acc_<name>.so
 (original code,        (CPU optimized)            |
  unchanged)                                  libsdr_acc_ov.so (OpenVINO)
                                                   |
                                              Intel NPU (NPU 5, AI Boost)
```

- `sdr_search_code()` (API unchanged) dispatches the parallel code search to
  the backend selected by `sdr_acc_init()`. The original implementation is
  kept unchanged as `sdr_search_code_cpu()` and is used for the CPU backend
  and as the fallback for shapes not supported by a backend.
- `sdr_search_code2()` adds `Nmax` (number of code offsets to compute).
  `search_sig()` uses only the code offsets `< N` of the `2N`-point
  correlation, so the only change of the acquisition logic is the call of
  `sdr_search_code2()` with `Nmax = N` (1 line in `sdr_ch.c`).
- The acquisition state machine, peak detection (`sdr_corr_max()`), C/N0
  estimation, fine Doppler estimation (`sdr_fine_dop()`) and the transition to
  tracking are not changed.
- Hardware/vendor-specific code is only in the plugin. The plugin interface
  (`src/sdr_acc.h`) uses plain tensors and has no dependency on the receiver
  data structures, so `libsdr` does not depend on OpenVINO. The plugin is
  loaded by `dlopen()` at runtime.

### 3.1 Backend selection

The backend is selected at runtime without source code changes by the option
`-acc backend` of `pocket_trk`, `pocket_acq` and `bench_sdr_acq`:

| backend | description |
|---------|-------------|
| `cpu` | CPU reference (original PocketSDR, default) |
| `cpu_opt` | CPU optimized: one DFT of IF data shared by all Doppler bins (shift of DFT) |
| `npu` | = `ov:NPU` |
| `ov:<device>[,direct]` | OpenVINO plugin on device `NPU`, `CPU`, ... `direct`: disable shared DFT (same algorithm as the CPU reference) |
| `<name>[:<opt>]` | plugin `libsdr_acc_<name>.so` with option string `opt` |

The plugin is searched in `$SDR_ACC_PATH`, the directory of the executable,
`<exe dir>/../lib/<os>` and the default library path. The compiled NPU models
are cached in `$SDR_ACC_CACHE` (default `~/.cache/pocket_sdr/ov`, empty: no
cache).

### 3.2 Plugin interface

```c
int search_code(const float *x, const float *code, int M, double fs,
    const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof);
```

One call is one non-coherent integration step of the parallel code search:

- `x [M][2]`: IF samples (complex float, before carrier wipe-off)
- `code [M][2]`: conjugate DFT of the code replica (as `sdr_gen_code_fft()`)
- `fc [K]`: carrier frequencies of the Doppler bins (IF + Doppler)
- `P [K][ldP]`: `P[k][n] += |IDFT(DFT(x * exp(-j2πfc[k]t)) * code)[n]|^2 / M^2`
  for `n < nout` (same scale as the CPU reference)
- returns 0 for unsupported shapes -> CPU reference fallback

`search_code_i8()` (plugin API version 3, optional) takes the IF samples as
complex int8 (for integer pipelines of NPUs); the generic layer uses it if
the plugin provides it.

The generic layer converts the IF ring buffer (`sdr_buff_t`, 4+4 bit complex)
to the float tensor and collects the statistics (`sdr_acc_stat()`): number of
calls/fallbacks, conversion, input, execution and output times and tensor
bytes.

## 4. Tensor Representation (OpenVINO plugin)

NPUs provide matrix engines but no FFT primitive. The `M`-point DFT
(`M = 2N`, `N` = samples per code period) is decomposed into dense complex
matrix products (four-step DFT) with `M = A * B` (`A <= B`, `A` max divisor of
`M` `<= sqrt(M)`):

```text
x[n1,n2] (n = B*n1+n2)
  -> F_A (left) -> twiddle W_M^(k1*n2) -> F_B (right) : X_0[k1+A*k2]
  -> Doppler shift X_k[m] = X_0[m+s_k] (row gather)
  -> * code[k1+A*k2]
  -> F_B^-1 (right) -> twiddle W_M^-(k1*p2) -> F_A^-1 (left, R rows)
  -> |c|^2 : P[k][B*p1+p2] (p1 < R = ceil(nout/B))
```

| signal (16 MHz) | M | A x B | Doppler bins K |
|-----------------|---|-------|----------------|
| 1 ms code (L1CA, B1I, G1CA, L5I, E5aI, E5bI, E6B) | 32000 | 160 x 200 | 21 |
| 4 ms code (E1B, L6D) | 128000 | 320 x 400 | 81 |

- Complex values are real tensors with re/im halves stacked along the
  contracted axis, so that each complex matrix product is one real MatMul.
- Shift mode (default): the Doppler bin step of PocketSDR (0.5 / T) equals the
  DFT bin `fs / M`, so the carrier wipe-off of bin `k` is a cyclic shift of the
  DFT of the first bin. The forward DFT is computed once per call and shared
  by all bins (same algorithm as `cpu_opt`). The carrier wipe-off of the first
  bin is done by the host. `direct` mode computes the carrier wipe-off and the
  forward DFT for each bin (same algorithm as the CPU reference).
- Precision: Intel NPU computes in FP16. The DFT matrices are unitary and the
  inputs are normalized to unit RMS to stay well inside the FP16 range. The
  output powers are FP16 and are scaled back to the CPU reference scale.
- Doppler chunks: the bins are processed in chunks of at most
  `MAX_CHUNK / M` bins (`MAX_CHUNK = 2688000`) per inference, because the NPU
  execution time increases superlinearly with the intermediate tensor size
  (`M = 128000`: K=21 8.1 ms, K=81 81 ms per inference).
- Static shapes: one model is compiled and cached per (`M`, `K`, `R`, shift).

## 5. Numerical Equivalence and Regression Tests

- `test/acc/golden_cpu.txt`: acquisition results of 309 signals/PRNs on the
  recorded IF data (FE 8CH, 16 MHz, L1/E1/B1/G1/G2/L5/E5a/E5b/L6/E6) generated
  by the original PocketSDR (before any change).
- `test/acc/cmp_acq.py`: compares a backend with the golden results.
  Tolerances: detection identical (except |C/N0 - threshold| < margin), code
  offset ±1 sample, fine Doppler ±0.1 bin step, C/N0 ±0.5 dB, peak power ±5%.
  The CPU backend must be bit-identical (hash of the correlation power map).
- `test/src/test_sdr_acc.c` (`make test`): CPU dispatcher identity, cpu_opt,
  plugins (`ov:CPU` FP32 and `npu` FP16) vs the CPU reference (NRMSE < 0.05,
  identical peak, C/N0 ±0.3 dB, fine Doppler), FP16 vs FP32 of the same graph
  (NRMSE < 0.005), CPU fallback of unsupported shapes and the Doppler chunks.

The differences of the backends from the CPU reference are dominated by the
quantization of the carrier LUT in the CPU reference (NRMSE ~1.5-2%), not by
FP16 (NRMSE 0.07% vs FP32 of the same graph).

## 6. Benchmark Procedure

```sh
# build (OpenVINO 2026.4 in ~/opt/openvino or $OV_DIR)
cd lib/build && make USE_SOAPY=0 USE_NPU=1 && make install USE_NPU=1
cd test/utest && make test bench_sdr_acq bench_sdr_acq_fftw

# regression + benchmark of all backends on recorded IF data
cd test/acc && ./bench_acq.sh [IF_file [out_dir [backend ...]]]
# on low-power cores (e.g. LP E-cores of Core Ultra)
TASKSET="taskset -c 12" ./bench_acq.sh "" out_lp
```

Rockchip RK3588 (section 13):

```sh
# models (PC with RKNN-Toolkit2 of the same version as the runtime)
cd lib/build && python3 gen_rknn_acq.py -f 32000:7 128000:27   # default (_f)
python3 gen_rknn_acq.py -w 32000:7 128000:27   # fallback (_w), bins not spaced by fs/M
python3 gen_rknn_acq.py -q 32000:7 128000:27   # INT8 (_fq, option int8)
# plugin (on the SBC, RKNN runtime in /usr or $RKNN_DIR)
cd lib/build && make USE_SOAPY=0 USE_RKNN=1 && make install USE_RKNN=1
cd test/acc && TASKSET="taskset -c 4" ./quick_acq.sh "" out rknn
```

Tracking (Phase 2):

```sh
cd test/acc && ./bench_trk.sh [IF_file [out_dir [backend ...]]]
# pocket_trk with batched tracking correlators
bin/pocket_trk ... -tacc cpu_mt file
```

- `bench_sdr_acq` runs the receiver channel state machine (`sdr_ch_update()`
  in SRCH state) for each signal/PRN on the same IF samples and measures the
  P-map computation of the first `T_ACQ` (20 ms): time per search, CPU time,
  CPU utilization, NPU utilization (`npu_busy_time_us` of the Intel NPU
  driver), package energy (RAPL, includes the NPU), and the backend
  statistics per call (conversion, input, execution, output, tensor bytes).
- `bench_sdr_acq_fftw` is the same benchmark linked with FFTW3 instead of
  PocketFFT. `-wisdom file` imports FFTW wisdom (generated with
  `FFTW_PATIENT` at the first run). Backends with prefix `fftw+` in
  `bench_acq.sh` use it.
- Baselines: the existing PocketSDR acquisition with PocketFFT (default build)
  and with FFTW3 (`USE_FFTW=1`). The NPU is compared both with the same
  algorithm (`ov:NPU,direct`) and with the shared DFT (`npu`); the shared DFT
  on the CPU (`fftw+cpu_opt`) is the counterpart of the latter.

## 7. Results (Phase 1)

Environment: Intel Core Ultra X7 358H (Panther Lake, 4 P + 8 E + 4 LP E
cores), Intel NPU 5 ("AI Boost", 3 tiles, FP16 25 TFLOPS / INT8 50 TOPS),
Linux 7.0, OpenVINO 2026.4.1, FFTW 3.3.10. IF data:
`L1B1G1L2G2L5E5bL6_20250327_0248_16MHz.bin` (FE 8CH, 16 MHz, 2 bit IQ), 309
signals/PRNs of 10 signal types. Single-threaded (one search at a time, as
`update_srch_ch()` of the receiver).

### 7.1 Numerical equivalence (all 309 signals, vs golden of original code)

| backend | detection | bit-exact | max diff C/N0 | code | fine Doppler | peak | result |
|---------|-----------|-----------|---------------|------|--------------|------|--------|
| cpu (PocketFFT) | 309/309 | 309 | 0 | 0 | 0 | 0 | PASS |
| fftw+cpu | 309/309 | 0 | 0.0 dB | 0 | 0.000 bin | 0.000 | PASS |
| npu | 309/309 | 0 | 0.3 dB | 0 | 0.041 bin | 3.7% | PASS |
| ov:NPU,direct | 309/309 | 0 | 0.3 dB | 0 | 0.042 bin | 3.7% | PASS |
| fftw+cpu_opt | 309/309 | 0 | 0.2 dB | 0 | 0.039 bin | 3.3% | PASS |

(C/N0 is printed with 0.1 dB resolution; differences of the NPU are dominated
by the carrier LUT quantization of the CPU reference.)

### 7.2 Acquisition time and energy per signal search (T_ACQ = 20 ms)

Time per search (ms) / CPU time (ms) / package energy (J). P: P-core,
LP: pinned to an LP E-core (`taskset -c 12`) as a proxy of a low-power CPU.

| backend | L1CA (1 ms code) P | L1CA LP | E1B (4 ms code) P | E1B LP |
|---------|--------------------|---------|-------------------|--------|
| cpu (PocketFFT) | 153.6 / 153.5 / 1.83 | 193.0 / 192.8 / 1.40 | 752.1 / 735.6 / 9.11 | 976.1 / 959.5 / 7.05 |
| fftw+cpu (FFTW PATIENT) | 36.5 / 36.3 / 0.51 | 81.6 / 81.4 / 0.61 | 199.5 / 183.0 / 2.68 | 439.7 / 423.1 / 3.14 |
| ov:NPU,direct (same algorithm) | 55.8 / 20.8 / 0.60 | 53.2 / 17.5 / 0.61 | 292.2 / 64.4 / 3.45 | 281.6 / 60.4 / 3.56 |
| **npu (shared DFT)** | **32.9 / 12.3 / 0.33** | **31.8 / 11.1 / 0.34** | **181.0 / 51.6 / 1.72** | **173.3 / 48.8 / 1.87** |
| ov:NPU,turbo | 24.8 / 24.6 / 0.34 | - | 140.0 / 137.8 / 2.07 | - |
| fftw+cpu_opt (shared DFT on CPU) | 19.2 / 19.1 / 0.27 | 44.8 / 44.6 / 0.32 | 117.7 / 101.0 / 1.49 | 240.1 / 223.5 / 1.70 |

The other 1 ms signals (B1I, G1CA, G2CA, L5I, E5aI, E5bI, E6B) are similar to
L1CA and L6D is similar to E1B (see `test/acc/out*/bench_*.txt`).

Summary:

- vs the existing acquisition (same algorithm): the NPU (`ov:NPU,direct`)
  is slower than FFTW on a P-core but faster on an LP core (L1CA 53 vs 82 ms,
  E1B 282 vs 440 ms). The CPU time is 1/4-1/7 of FFTW.
- With the shared DFT on both sides, the NPU (`npu`) is slower than the CPU
  (`fftw+cpu_opt`) on a P-core, and faster on an LP core (L1CA 32 vs 45 ms,
  E1B 173 vs 240 ms) with the CPU time of 1/4-1/5.
- The NPU time does not depend on the CPU, so the relative advantage of the
  NPU increases on a weaker CPU. The energy per search of the NPU is lower
  than FFTW and similar to `fftw+cpu_opt`.
- Per call of the NPU backend (L1CA, 21 bins): execution 1.2-1.5 ms, input
  preparation 0.15 ms, output accumulation 0.14 ms; (E1B, 81 bins in 4
  chunks): execution 29 ms, output accumulation 7 ms.

### 7.3 End-to-end receiver (pocket_trk, 95 s real-time playback, 10 signal types)

| backend | SIGNAL FOUND | unique signals | re-acquisitions | SIGNAL LOST |
|---------|--------------|----------------|-----------------|-------------|
| cpu | 75 | 75 | 0 | 8 |
| npu | 83 | 74 | 9 | 11 |

All SIGNAL LOST events of both backends occur exactly 2.0 s after the
acquisition (first C/N0 and PLI decision window) of weak signals near the
acquisition threshold (acquisition C/N0 30-39 dB-Hz: L5I 27, E5aI 18, E5bI 18,
...). The faster NPU search re-acquires such signals more often within the
same time, which increases both FOUND and LOST counts. The number of tracked
signals is the same; the difference is not caused by the numerical accuracy
of the NPU backend.

## 8. Optimization Notes (Intel NPU, OpenVINO 2026.4)

What was tried in Phase 1 and the observations (L1CA: M=32000, K=21; E1B:
M=128000, K=81; execution time per inference):

| change | result | adopted |
|--------|--------|---------|
| Doppler chunks (E1B: 4 x 21 bins) | E1B 79 -> 29 ms | yes |
| host carrier wipe-off in shift mode (~20 fewer nodes) | fixed cost 1.4 -> 0.4 ms, L1CA 1.9 -> 1.7 ms | yes |
| NPU turbo mode (`NPU_TURBO`) | L1CA 1.7 -> 0.9-1.1 ms, E1B 29 -> 23 ms, but the driver keeps the CPU busy (CPU utilization ~90-97%) | no (removed, conflicts with freeing the CPU for tracking) |
| INT8 (FakeQuantize on MatMul inputs) | E1B -15%, L1CA +75% (FQ nodes), NRMSE 3-5% | no |
| FP16 input tensors | no change of execution, slower host conversion | no |
| parallel infer requests (chunks) | no speedup (inferences are serialized) | no |
| sleep before waiting the completion | the NPU/CPU clock down, 1.7 -> 4-7 ms | no |
| graph layouts without inner-axis split/concat ([re\|im] columns, 2D MatMul only) | slower (2.7-3.4 ms) in spite of fewer copies | no |
| non-coherent accumulation on the NPU (powers kept in FP16 device tensors until a flush before reading P) | L1CA -6% per call (within the run-to-run variation of the receiver benchmark, no reduction of CPU time), E1B slower (+5 ms per call, +7 ms per flush) | no (reverted) |
| streaming loads (MOVNTDQA) for the output tensors | no change | no |
| direct mode: carrier outer product a_k[n1] * b_k[n2] by MatMul (inner dimension 2) instead of broadcast complex multiply | L2CM-size graph 66.7 -> 55.0 ms per inference (-17%) | yes |
| min 8 Doppler bins per inference (MIN_CHUNK) | M=640000: 4 -> 8 bins 1795 -> 1366 ms per call (-24%); M=320000 unchanged (8) | yes |
| direct mode: IF samples set once per call (not per chunk) | host input time of large M reduced | yes |
| direct mode: tiled IF samples or operand order of the carrier wipe-off | no improvement (+0-15%) | no |
| host CPU time: code DFT kept in the input tensor while the same code is searched (pointer + signature), RMS of IF samples by subsamples (scale of FP16 range only, exactly restored) | direct mode L1CA 59.1 -> 46.4 ms / CPU 20.6 -> 14.4 ms per search, E1B CPU 70.5 -> 56.9 ms (waiting for the NPU completion is only ~5-10% of the execution time, the CPU time was dominated by the RMS of IF and code) | yes |
| pipelined chunks by 2 infer requests (host input of chunk j and output of chunk j-1 during NPU execution) | per call (3 interleaved runs): E1B 49.3 -> 45.1 ms (-8.5%), L1C 493 -> 444 ms (-10%), L1CA (1 chunk) unchanged; but CPU time per call E1B 7.9 -> 9.0-10.1 ms (+15-27%: memory contention of host input/output with the NPU (+0.7 ms) and async execution) | option `pipe` (to be evaluated on the target SBC) |
| L1CA split into 2-3 chunks for pipelining | no improvement (fixed cost per inference) | no |
| direct mode E1B: DFT factorization and Doppler bins per inference | 320x400/21 bins 46.7 ms; 400x320 or 500x256 with 11-41 bins 42.0-43.2 ms (-9-10%); 320x400/11 bins 44.6 ms (-4.5%) | no (platform dependent, to be tuned on the target NPU) |
| int8 IF samples input (plugin API version 3 `search_code_i8()`, scale folded into the carrier a_k, direct mode) | no significant change (L1CA 48.8 -> 47.7 ms, E1B 278-287 -> 275-278 ms, host input -0.04 ms per call; the IF conversion cost is the LUT access, not the output type) | yes (API kept for integer pipelines on SBC NPUs) |
| OpenVINO DFT/IDFT ops on NPU (instead of MatMul DFT) | 21 x (DFT+IDFT) of 32000 points 165 ms (compile 139 s) vs 1-3 ms of the MatMul DFT search | no |
| time-domain correlation by grouped Conv1D (kernel = code, 16000 taps) | K=1: 118 ms (4.3 GMAC/s); K=21: compile not finished in 15 min; 30x MACs of the DFT search | no |
| carrier by Sin/Cos ops on NPU (K x M) | L1CA size 1.1-1.6 ms, E1B size 3.9-5.4 ms vs separable outer product MatMul 0.38 / 1.15 ms (and FP16 phase precision issue) | no |
| DFT factorization M = A x B (row/column swap and other divisors) | M=32000: 160x200 (current) fastest (200x160 +23%, 8 others slower); M=128000: 500x256 24.0-24.3 ms vs 320x400 27.3-28.0 ms (-12%, reproduced twice), 400x320 -6%, 512x250 +35% | no (to be tuned on the target NPU) |

Observations:

- The best DFT factorization depends on the tiling of the NPU compiler, not
  on the signal: it is a speed-only parameter (identical DFT, FP16 rounding
  only) determined by M and the Doppler bins per inference, and it may change
  with the NPU generation, driver and compiler versions. It should be tuned
  (e.g. measured at the first compilation and cached) on the target NPU.

- Each graph node has a fixed cost of ~30-90 us. With ~50 nodes, the fixed
  cost of an inference is ~1.3 ms, which dominates the 1 ms code search.
- MatMuls are efficient (e.g. 16 GMAC in 3.4 ms), but element-wise ops on
  large tensors are slow, especially on strided inputs (split along the
  innermost axis). The node times depend on the layouts chosen by the NPU
  compiler and are hard to predict. The achieved throughput is ~5% of the
  FP16 peak (25 TFLOPS).
- The P-map accumulation on the host (`P += |c|^2`, 20 MB for E1B) is memory
  bound (~7 ms per call for E1B).

## 8.0 Direct Acquisition of Long Codes (L1C 10 ms, L2CM 20 ms)

PocketSDR acquires the codes longer than 4 ms (`MAX_ACQ`) only by assisted
acquisition (Doppler of a tracked signal of the same satellite, 3 bins,
100 ms). The direct (unassisted) search of the long codes (e.g. weak signals
where only the pilot L1CP can be acquired, or L1 band interference for L2C)
needs fine Doppler bins (0.5/T): L1C 201 bins (M=320000), L2CM 401 bins
(M=640000), ~10x and ~19x of the L1CA search. Time / CPU time / energy per
search without the shared DFT on both sides (`test/acc/long_sigs.sh`, 11 GPS
PRNs, detections identical to the CPU reference):

| | FFTW (existing) | NPU direct (before) | NPU direct (improved) |
|---|---|---|---|
| L1C P-core | 610 ms / 590 / 8.3 J | 1390 ms / 469 / 14.9 J | 1188 ms / 275 / 12.5 J |
| L1C LP E-core | 1195 ms / 1175 / 8.9 J | - | 1119 ms / 198 / 12.3 J |
| L2CM P-core | 1442 ms / 1421 / 19.5 J | 2772 ms / 1234 / 32.5 J | 1950 ms / 854 / 23.1 J |
| L2CM LP E-core | 3124 ms / 3104 / 22.9 J | - | 1605 ms / 526 / 21.7 J |

On an LP E-core the NPU is 1.07x (L1C) and 1.95x (L2CM) faster than FFTW with
1/6 of the CPU time. The remaining cost of the direct mode is the element-wise
carrier wipe-off of each Doppler bin (K x M), which only the shared DFT
removes (expected ~0.6x of the time on both CPU and NPU).

## 8.1 Integer Implementation of the Code Search (INT8 / INT16)

For NPUs without floating point (or faster in integer), the code search was
evaluated with integer quantization emulated exactly (rounding and clamping)
in the same graph on the OpenVINO CPU device (plugin options `q8`, `q16`,
`qa=b0-b1-b2-b3`, `qw=b`, `clip=k`; e.g. `-acc ov:CPU,q16,clip=8`):

- The inputs of the 4 MatMuls (DFT n1, n2 and IDFT k2, k1) and the constants
  (DFT matrices, twiddles, code DFT) are quantized; the MatMul outputs are
  accumulators (int32) and the correlation output is not quantized.
- Scales are analytic: the IF data are normalized to unit RMS and the DFTs
  are unitary, so the MatMul inputs are ~Gaussian with sigma 1/sqrt(2) (IF
  data are noise-dominated) and clipped at +/- clip * sigma.
- The code DFT is normalized by its max magnitude (not RMS): the code spectrum
  is concentrated in the main lobe (sinc^2 envelope of the chips), so with the
  RMS normalization the products z * code were clipped (NRMSE 5.5% at the
  first IDFT stage). With |code| <= 1, |z * code| <= |z| and the same range
  bounds all stages.

| configuration | synthetic NRMSE vs FP32 | 309 signals vs golden (C/N0, code, fine Doppler, peak) | result |
|---------------|-------------------------|---------------------------------------------------------|--------|
| FP32 (same graph) | - | 0.3 dB, 0, 0.041 bin, 3.7% | PASS |
| INT16 all (clip 8 sigma) | 0.034% | 0.3 dB, 0, 0.041 bin, 3.7% | PASS |
| INT8 DFT / INT16 IDFT inputs, INT8 constants (clip 8) | 2.5% | 0.2 dB, 0, 0.041 bin, 4.4% | PASS |
| INT8 all (clip 4 sigma) | 3.7% | 0.4 dB, 1 sample, 0.034 bin, 4.8% | PASS (small margin) |

Speed on Intel NPU (quantization as FakeQuantize, converted to INT8 kernels
by the NPU compiler; NRMSE on the NPU same as the emulation): direct mode
L1CA 3.04 -> 5.23 ms (slower, the FakeQuantize nodes add the fixed cost per
node), E1B 48.9 -> 40.3 ms (-18%). The NPU has no INT16 kernels. The INT8
speed should be evaluated on the target NPU with integer MatMul APIs (e.g.
RKNN int8 x int8 -> int32) without the quantization nodes.

INT16 is equivalent to floating point. INT8 passes with small margins (peak
power within 5%); the remaining error is the quantization of the products
z * code with the non-flat code spectrum. A possible improvement is a
per-frequency-bin scale of the products folded into the IDFT matrix (weights).

## 9. Current Limitations

- Signal/channel-wise acquisition (one PRN per call, no PRN batch) as
  specified for Phase 1. The receiver searches one channel at a time
  (`update_srch_ch()`), so the NPU is idle most of the time in real-time
  operation.
- The non-coherent integration is accumulated on the host per call.
  Accumulation on the NPU needs a sync point in `search_sig()` (not done).
- Intel NPU via OpenVINO and Rockchip NPU via RKNN (section 13) are
  implemented. Static shapes; the first compilation of an OpenVINO model takes
  ~0.5 s (cached afterwards). The RKNN models are precompiled per shape on a
  PC (RKNN-Toolkit2); shapes without a model use the RKNN matmul API.
- The CPU usage of the NPU backend includes the busy polling of the NPU
  driver while waiting for the completion.
- Rockchip RK3588 (section 13):

```sh
# models (PC with RKNN-Toolkit2 of the same version as the runtime)
cd lib/build && python3 gen_rknn_acq.py -f 32000:7 128000:27   # default (_f)
python3 gen_rknn_acq.py -w 32000:7 128000:27   # fallback (_w), bins not spaced by fs/M
python3 gen_rknn_acq.py -q 32000:7 128000:27   # INT8 (_fq, option int8)
# plugin (on the SBC, RKNN runtime in /usr or $RKNN_DIR)
cd lib/build && make USE_SOAPY=0 USE_RKNN=1 && make install USE_RKNN=1
cd test/acc && TASKSET="taskset -c 4" ./quick_acq.sh "" out rknn
```

Tracking (Phase 2): the NPU tracking backend is functional and verified
  but slower than the CPU on this platform (section 11), so `pocket_trk -tacc
  npu` cannot run in real time. Batched correlators exclude L6D/L6E and E5ABQ.
- Sections 7-12 were measured on an x86 laptop SoC (the LP E-cores as a
  proxy of a low-power CPU); section 13 on RK3588 (ROCK 5B).

## 10. Supported NPU Backends (Acquisition and Tracking)

| backend | plugin | device | status |
|---------|--------|--------|--------|
| Intel NPU (Core Ultra, NPU 5) | `libsdr_acc_ov.so` (OpenVINO 2026.4) | `NPU` | implemented, verified |
| OpenVINO CPU (FP32, reference of the graph) | `libsdr_acc_ov.so` | `CPU` | implemented (test only) |
| Rockchip RK3588 NPU (acquisition) | `libsdr_acc_rknn.so` (RKNN runtime 2.3.0, models by RKNN-Toolkit2 2.3.0) | `rknn` | implemented, verified (section 13) |
| AMD XDNA, Apple ANE, Qualcomm Hexagon | - | - | future (same plugin interface, own low-level implementation) |

Setup of the Intel NPU (Linux): the NPU device `/dev/accel/accel0` must be
accessible by the user (group `render` or ACL). OpenVINO is installed from the
archive (e.g. `~/opt/openvino`, or set `OV_DIR`).

## 11. Phase 2: Batched Tracking Correlators

### 11.1 Structure

The tracking of a channel (`track_sig()` in `sdr_ch.c`) is separated into
three steps without changing the algorithm:

```text
sdr_ch_update_pre()    state prediction (carrier phase, code offset,
   |                   adj_coff()) and correlator job (sdr_corr_job_t)
   v
sdr_acc_corr_batch()   batched standard correlators of the jobs
   |                   (carrier wipe-off + correlation) by tracking backend
   v
sdr_ch_update_post()   P history, secondary code, FLL/PLL/DLL, C/N0,
                       navigation decoding, loss-of-lock
```

`track_sig()` itself is the same composition (`track_pred()`, `track_job()`
+ `sdr_corr_std()`, `track_update()`), so `sdr_ch_update()` is unchanged.
L6D/L6E (CSK) and E5ABQ (AltBOC) with special correlators are updated by
`sdr_ch_update()` in `sdr_ch_update_pre()` and are not batched.

Receiver scheduling (`sdr_rcv.c`, option `trk_batch`, `pocket_trk -tacc
backend`): the channels are grouped by RF channel and code period (samples
per correlation). A group thread updates the tracking channels of the group
epoch by epoch (one batch per epoch, all jobs share the same IF samples). The
search and idle states stay in the channel threads (a search takes 30-750 ms
and must not stall the tracking of the group). The epoch ownership is passed
by the channel lock (`sdr_ch_th_t.mtx`); a channel locked by the search is
caught up by the group thread to the epoch of the group. Without `-tacc`, the
original channel threads are used.

Tracking backends (`sdr_acc_trk_init()`):

| backend | description |
|---------|-------------|
| `cpu` | `sdr_corr_std()` for each job (reference) |
| `cpu_mt[:n]` | jobs of a batch by a thread pool of n threads (default: CPU cores) |
| `npu`, `ov:<device>` | OpenVINO plugin `corr_batch()` (plugin API version 2) |

Plugin interface (`sdr_acc.h`): the IF samples of the batch (complex int8,
shared by all channels), and for each channel the carrier (fc, phase), the
code scale and for each correlator the resampled code slice and the code
wrap-around sample. The plugin returns the correlations before and after the
code wrap-around; the generic layer applies the polarity detection and the
normalization of `sdr_corr_std2()`.

OpenVINO graph: carrier `a_c[b] * b_c[t]` (separable, `s = L*b + t`,
computed by host in double), carrier wipe-off on the NPU, code replicas
rotated by host (2 rows per correlator: before/after wrap-around, int8), and
one batched MatMul `[C,2n,N] x [C,N,2]`. Channels are padded to 16 * 2^k.

### 11.2 Verification

- `bench_sdr_trk` (deterministic, no threads): acquisition and 3 s tracking
  of 33 channels (L1CA, E1B, B1I) on the recorded IF data; per-epoch tracking
  states and correlator outputs. `cmp_trk.py` compares them.
- The original `track_sig()` (HEAD), the refactored `sdr_ch_update()`, the
  batched `cpu` and `cpu_mt` backends: **bit-identical** (112422
  channel-epochs).
- `npu` vs the CPU reference: correlator outputs relative error median 1.0%
  (p95 2.5%) (carrier LUT quantization of the CPU reference), DLL
  discriminator diff median 0.0016, code offset max 0.33 ns, Doppler p99
  0.066 Hz, ADR max 0.010 cyc, C/N0 max 0.03 dB on 29 stable channels. 2 weak
  channels near the thresholds changed the loss-of-lock decision (one in each
  direction). PASS.
- `test_sdr_acc`: batched correlators across the IF buffer boundary and
  different IF blocks (CPU backends identical to `sdr_corr_std()`, plugins
  within tolerance).
- `pocket_trk` 65 s real-time, 10 signal types: original 51 found / 5 lost /
  48 PVT epochs, `-tacc cpu` 52 / 4 / 48, `-tacc cpu_mt` 47 / 1 / 47 (PVT
  solutions within a few m). `-tacc npu` cannot keep up with real time (see
  below): 27 / 15 / 0.

### 11.3 Scaling (bench_sdr_corr)

Time per channel and epoch (us), L1CA (N=16000, 1 ms, real-time budget
1 ms per batch), random IF samples:

| backend | 4 taps | 11 | 21 | 51 | 105 |
|---------|--------|----|----|----|-----|
| cpu (1 P-core) | 6.2 | 13.6 | 24.7 | 39.6 | 79.2 |
| cpu_mt:16 | 0.74 | 1.6 | 2.9 | 5.2 | 10.6 |
| cpu (1 LP E-core) | 14.0 | - | 54.9 | - | - |
| cpu_mt:4 (4 LP E-cores) | 4.1 | - | 15.5 | - | - |
| npu (256 ch) | 67 | 157 | 437 | 2467 | 10357 |

Max channels in real time (4 taps): cpu 1 P-core ~160, cpu_mt:16 ~1300,
cpu_mt:4 LP ~250, npu < 16 (16 ch: 1.4 ms per batch). E1B (N=64000, 4 ms):
cpu 24-29 us/ch, cpu_mt:16 3.0 us/ch, npu 217-491 us/ch.

NPU breakdown (256 ch, 4 taps, per batch): host input preparation 5.4 ms
(replica rotation 128 kB/ch), NPU execution 11.6 ms (MatMul 2.9 ms, int8 to
FP16 conversion of replicas 1.9 ms, carrier wipe-off ~3 ms).

### 11.4 Findings

- The per-epoch tracking correlation has no data reuse except the IF samples:
  each channel and correlator needs its own replica (C x n x N) and the
  carrier wipe-off is C x N. It is a batched GEMV bound by the memory traffic
  and the element-wise operations, which the NPU executes much slower than
  the int8 SIMD (AVX2 `maddubs`) of the CPU with the fused LUT carrier mixing.
- Tried: INT8 MatMul (FakeQuantize, no replica conversion): NPU execution
  10.5 -> 7.4 ms (256 ch), still 5x slower than one CPU core. Wide
  correlators by a Hankel MatMul (all 32 lags x 10 sub-sample code phases per
  channel): 130 us/ch, slower than the CPU for 105 taps (the MatMul has only
  10 output columns).
- On this platform the batched CPU backend (`cpu_mt`) is the effective
  tracking backend: the batch structure lets the correlators of all channels
  of an epoch be distributed over the cores (~1300 channels on 16 cores).
- The NPU is useful for the acquisition (FFT-like dense DFT matrices), and
  the tracking would need correlation work with data reuse to benefit from it
  (e.g. accumulating multiple epochs against the same replica, which is beyond
  the original tracking algorithm kept in Phase 2).

## 12. Decimation (Channelizer) for Tracking and Acquisition

The narrowband signals (e.g. L1CA BPSK(1), main lobe +/-1.023 MHz) are
oversampled at 16-24 MHz, which is needed for the wideband signals (L5, E5a/b
+/-10.23 MHz). A channelizer (FIR lowpass + decimation per band) reduces the
sampling rate of the narrowband signals for tracking. The FIR was implemented
on the CPU (AVX2 polyphase, only output samples computed) and on the NPU
(OpenVINO strided GroupConvolution, int8 input, 20 ms blocks)
(`test/acc/fir_dec.cpp`, `test/acc/ana_trk_dec.py`).

### 12.1 FIR on CPU vs NPU (L1 RF CH, 16 MHz -> 4 MHz, 64 taps, 4 s)

| | CPU (AVX2) | NPU (strided conv) |
|---|---|---|
| P-core: throughput / CPU load | 591 Msps / 2.7% of a core | 52 Msps / 5.5% of a core |
| LP E-core: throughput / CPU load | 379 Msps / 4.2% of a core | 55 Msps / 3.5% of a core |
| output | reference | NRMSE 0.044% |

The FIR is cheap on the CPU (3-4% of a core per band); the NPU (26% of the
signal time on the NPU, and host copies) does not reduce the CPU load.

### 12.2 Tracking at decimated rates (L1CA, 7 stable channels, 3 s)

| input | cost per channel-epoch | C/N0 | code noise | Doppler jitter |
|---|---|---|---|---|
| 16 MHz (original) | 7.54 us | ref | ref (1.4-3.1 cm) | ref |
| 8 MHz (fc 3.5 MHz) | 5.03 us (1.5x) | -0.12 dB | x1.06 | x1.09 |
| 4 MHz (fc 1.6 MHz) | 3.88 us (1.9x) | -0.53 dB | x2.4 | x1.06 |
| 4 MHz (fc 1.2 MHz) | 3.43 us (2.2x) | -0.54 dB | x1.9 | x1.14 (1 lost) |

The cost is not proportional to the rate (fixed ~2.5 us per channel-epoch of
loops etc.). 8 MHz is almost lossless for L1CA.

### 12.3 Acquisition at decimated rates (without shared DFT, after model cache)

| | FFTW P-core | NPU direct P-core | FFTW LP E-core | NPU direct LP E-core |
|---|---|---|---|---|
| L1CA 4 MHz (M=8000) | 8.2 ms / CPU 8.0 | 27-28 ms / CPU 8.5-9.6 | 18.8 ms | 33.9 ms |
| L1CA 8 MHz (M=16000) | 28.1 ms / CPU 27.9 | 26.2 ms / CPU 11.3 | 52.8 ms | 34.9 ms |
| E1B 8 MHz (M=64000) | 135 ms / CPU 118 | 134 ms / CPU 24.8 | 253 ms | 131 ms |

The NPU has a floor of ~1.3 ms per inference of the code search graph, so for
small DFTs the CPU FFT is faster (L1CA at 4 MHz: 0.4 ms per call by FFTW).
Batching PRNs at small M does not change it (M=8000: 1.35 ms per PRN with 8
PRNs per inference).

### 12.4 Findings

- The decimation (CPU, 3-4% of a core per band) increases the tracking
  capacity of narrowband signals by 1.5x (8 MHz, almost lossless) to 1.9x
  (4 MHz, -0.5 dB) independently of the NPU.
- With decimated narrowband signals, the CPU FFT acquisition of short codes
  becomes cheap and the NPU loses (4 MHz). The NPU acquisition remains
  effective for larger DFTs: wideband signals at the full rate (L5, E5a/b),
  4 ms codes (E1B 8 MHz: same time with 1/4.8 CPU on a P-core, 1.9x faster on
  an LP core) and the direct search of long codes (section 8.0).
- No NPU-specific method to reduce the per-channel tracking cost was found
  (correlator: section 11, FIR: 12.1, Conv/Sin/Cos/DFT ops: section 8).

## 13. Rockchip RK3588 (ROCK 5B)

### 13.1 Platform and measurement conditions

- Radxa ROCK 5B (RK3588: 4x Cortex-A76 2.3 GHz + 4x Cortex-A55 1.8 GHz,
  NPU 3 cores, 8 GB LPDDR4x), heatsink (13.1-13.7) and heatsink with fan
  (13.8-), Radxa Debian 12, kernel
  6.1, RKNPU driver 0.9.8, RKNN runtime (librknnrt) 2.3.0. Desktop stopped
  (multi-user.target). Built natively with `-mcpu=native` (NEON).
- CPU governor `performance`, and the devfreq governors of the DDR (`dmc`)
  and the NPU set to `performance`. With the default `dmc_ondemand`, the DDR
  stays at 528 MHz (max 2112 MHz) and the memory-bound steps vary by 2x (e.g.
  E1B FFTW search 1614 ms vs 1394 ms with `performance`). All results below
  are with `performance` (not persistent):

```sh
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
echo performance | sudo tee /sys/class/devfreq/dmc/governor
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

- Single-core results are pinned by `taskset` (A76: core 4, A55: core 0). The
  temperature stayed below 69 C (SoC) and the clocks never dropped.
- Recorded IF data: the data set of section 5; the first 10 s on the SD
  card (13.1-13.7, benches), the full 300 s from a NAS (13.8-13.9, real-time
  replay at 64 MB/s, see 13.8).

### 13.2 Plugin (libsdr_acc_rknn.so)

`src/sdr_acc_rknn.cpp`, built by `lib/build/libsdr_acc_rknn.mk` (`make
USE_RKNN=1`), selected by `-acc rknn[:opt]`. The Doppler bins of a call are
split into chunks for the 3 NPU cores; each chunk is processed by a worker
thread with its own RKNN context bound to one NPU core (the multi-core mode
of one context, `RKNN_NPU_CORE_0_1_2`, is not faster: 19.1 -> 17.3 ms for 21
bins, while 3 contexts in parallel take 9.6 ms). Two implementations:

1. **RKNN model** (default): the whole code search of a chunk is one
   inference of a model precompiled by RKNN-Toolkit2 on a PC
   (`lib/build/gen_rknn_acq.py`, models `lib/linux/rknn/acq_M<M>_K<Kc>_R<R><kind>.rknn`,
   searched in `$SDR_ACC_RKNN_MODEL` or `<plugin dir>/rknn`). Same algorithm
   as the CPU reference (carrier wipe-off and DFT for each bin, no shared
   DFT). Model kinds (searched in this order):
   - `_fq` (option `int8`): INT8 of `_f` (section 13.5).
   - `_s` (default, `gen_rknn_acq.py -s`, option `noshift`: not used): shared
     DFT (shift mode, section 13.13). Input: IF samples wiped by the carrier of
     the first bin of the chunk, the code DFT of bin k rolled by k bins.
   - `_f` (`gen_rknn_acq.py -f`): carrier folded into the DFT
     (section 13.3). Input: IF samples wiped by the carrier of the first bin
     of the chunk (host), the code DFT replicated along W. Needs bins spaced
     by fs / M (PocketSDR: 0.5 / T); otherwise `_w` is used.
   - `_w` (`-w`, option `nofold`): carrier wipe-off in the model (inputs: IF
     samples shared by the bins and the separable carrier a[n1], b[n2] of each
     bin).
   - no suffix (option `hostwipe`): carrier wipe-off by the host (NEON FP16).
2. **matmul API** (option `mm`, and fallback if no model file): the 4 DFT
   stages are `rknn_matmul` calls (FP16 x FP16 -> FP16, "data x DFT matrix",
   bins stacked along the rows), and the element-wise steps (carrier,
   twiddle + transpose, code product, power) are done by the host with NEON
   between the calls. Works for any shape without a model, but the host steps
   are bound by the memory traffic (5 passes over Kc x M complex values).

Model graph (ONNX, opset 13, FP16 on the NPU): the RKNN compiler maps a MatMul
to a 1x1 Conv over the channels and inserts Transposes around each MatMul
(first prototype: 6 Transposes = 57% of 12.2 ms). The graph is therefore
written in the NCHW layout of the NPU: each DFT stage is a 1x1 Conv over the
channels (complex: 4 real Conv + Add/Sub, no channel split/concat), and the
axis change between the stages is 2 Transposes (n1 -> n2 and k2 -> k1; the
forward F_B output k2 is already the channel axis of the IDFT F_B^-1):

```text
xr, xi [Kc][n1][1][n2]  -> Conv F_A -> twiddle -> Transpose -> Conv F_B
  -> * code [1][k2][1][k1] -> Conv F_B^-1 -> twiddle -> Transpose
  -> Conv F_A^-1 (R rows) -> P [Kc][p1][1][p2] = |c[B*p1+p2]|^2 / 64
```

- I/O by zero-copy in the native layout of the NPU (NC1HWC2, C2 = 8, FP16,
  `RKNN_QUERY_NATIVE_INPUT_ATTR`, `pass_through`), written/read by the host
  directly: host CPU per inference 0.19 ms. Inputs with one channel (the
  carrier b[n2], native NHWC) are passed in the normal FP16 NHWC layout (the
  native NHWC with a padded W stride gave wrong results). With
  `rknn_inputs_set()` (FP32
  NCHW) the conversion takes 4.2 ms of CPU, with zero-copy FP16 NHWC 5.2 ms
  (inside `rknn_run()`).
- NPU time of one inference (1 core, L1CA M=32000 Kc=7 / E1B M=128000 Kc=27):
  host wipe-off model 6.3 / 115 ms, model with wipe-off 7.9 / 137 ms.
  Breakdown (L1CA): Transpose 37%, Conv 32%, Mul 20%, Add/Sub 10%. The n2
  axis as H instead of W is slower (12.2 ms).
- The output power is scaled by 1/64 in the model to stay in the FP16 range
  (|c|^2 up to M for unit-RMS input).
- Chunks: the bins per inference Kc are limited by Kc x A <= 16384 (the
  first Conv has Kc x A output channels; a model for the L2CM direct search
  in one chunk per core, M=640000 Kc=134, needs 864 MB of NPU memory per
  context and fails to allocate the IOVA), and the chunks are processed in
  rounds of the 3 NPU cores (e.g. L1CP 201 bins: 9 chunks of 23, L2CM 401
  bins: 21 chunks of 20). The DFT factorization M = A x B takes the largest
  A <= sqrt(M) with B % 4 = 0 (M=320000: 500 x 640; with 512 x 625 the
  native layout of the odd width gave a wrong correlation, peak -4.6 dB).
- Models used by `pocket_trk` and the benches (16 MHz): 32000:7, 128000:27
  (full search of 1 / 4 ms codes, 21 / 81 bins), 32000:29 (85 bins), 8000:7
  (4 MHz, `test_sdr_acc`), 320000:23, 640000:20 (direct search of L1C / L2CM,
  section 13.10), and Kc = 1 for M = 32000, 128000, 320000, 640000 (assisted
  acquisition, 3 bins: one bin per core; without them the assisted searches of
  the receiver used the matmul API, which costs as much CPU as the FFT).
- RKNN matmul API: FP16 400 GFLOPS per NPU core with an explicit core mask
  (190 GFLOPS with the default auto mode), ~0.1 ms per call. The native
  layout conversion of B (`rknn_B_normal_layout_to_native_layout()`) does not
  support the FP16 output type (converted with the FP32 output type, same B
  layout).

### 13.3 NPU-internal optimization of the model

NPU time per inference (L1CA M=32000 Kc=7 / E1B M=128000 Kc=27, FP16, 1 NPU
core and 3 cores in parallel = all bins of a call):

| model | L1CA 1 core | L1CA 3 cores (21 bins) | E1B 1 core | E1B 3 cores (81 bins) |
|-------|-------------|------------------------|------------|-----------------------|
| `_w`: wipe-off in the model (previous default) | 7.9 | 11.7 | 137 | 200 |
| + carrier folded into F_A weights and twiddles | 6.1 | 9.2 | 112 | 162 |
| + bins in channels/width (batch 1) | 5.0 | 7.2 | (W > 8192: CPU fallback) | (Kc=14 x 2: 159) |
| + complex Conv by 3 real Conv (= `_f`) | **4.9** | **7.2** | **105** | **156** |
| `_fq`: INT8 (F_A^-1 and power in FP16) | 3.5 | 5.0 | 65 | 95 |

- Profile of `_w` (L1CA): Mul 32%, Transpose 30%, Conv 26%, Add/Sub 13%. The
  element-wise ops (33 per inference, 0.1-0.18 ms each) are bound by the DDR
  bandwidth of the NPU; the broadcast Muls of the carrier wipe-off (12 ops)
  were the most expensive.
- **Carrier folding**: the PocketSDR Doppler bins are spaced by the DFT bin
  fs / M, so carr_k[n] = carr_0[n] W_M^(k n) and with n = B n1 + n2,
  W_M^(k n) = W_A^(k n1) W_M^(k n2). The factor W_A^(k n1) is folded into the
  F_A weights of bin k (rows rotated by k; the Kc bins are the output channels
  of one Conv on the shared input) and W_M^(k n2) into a per-bin twiddle
  constant. The host only wipes the IF samples with the carrier of the first
  bin of the chunk (M samples). Each bin still has its full DFT on the NPU
  (same MACs, no shared DFT); the 12 element-wise ops of the wipe-off are
  removed.
- **Batch 1**: a Transpose of [7,160,1,200] takes 0.64 ms, of the same data as
  [1,1120,1,200] 0.37 ms. Keeping the bins in the channels/width ([1][(k,k1)][1][n2]
  -> [1][n2][1][(k,k1)]) halves the Transpose time; the reshape to [Kc,A,1,B]
  before the last stage is free for A % 8 = 0. The width is limited to 8192
  (larger: CPU fallback), so E1B (Kc x A = 8640) keeps the bins in the batch.
- **3 real Conv** (Wr(Xr+Xi), (Wr+Wi)Xi, (Wi-Wr)Xr): -3% (L1CA) / -6% (E1B).
- Tried and rejected: prime-factor DFT (no twiddles, M = 125 x 256): slower
  (7.8 ms, channel count 125 not aligned, Reshape and Transposes 4.9 ms);
  grouped Conv for the last stage instead of the reshape (5.5 ms); bins in
  H x W for the last stage (5.7 ms); n2 in H instead of W (12.2 ms).

### 13.4 Acquisition results

Time per signal search (T_ACQ = 20 ms) / CPU time per search (ms), 30
signals of `quick_acq.sh` (all PASS vs golden), single core:

| backend | L1CA A76 | E1B A76 | L1CA A55 | E1B A55 |
|---------|----------|---------|----------|---------|
| cpu (PocketFFT) | 437 / 437 | 2094 / 2078 | 1713 / 1713 | 9466 / 9450 |
| fftw+cpu (FFTW PATIENT) | 252 / 252 | 1394 / 1378 | 1119 / 1119 | 5541 / 5525 |
| rknn:mm (matmul API) | 163 / 159 | 736 / 601 | 533 / 528 | 2372 / 2272 |
| rknn:hostwipe (model) | 218 / 42 | 977 / 195 | 320 / 185 | 1415 / 665 |
| rknn:nofold (model `_w`) | 257 / 31 | 1098 / 120 | 334 / 116 | 1398 / 470 |
| **rknn (model `_f`, default)** | **167 / 31** | **914 / 161** | **231 / 102** | **1114 / 394** |
| rknn:int8 (model `_fq`) | 121 / 29 | 584 / 137 | 191 / 113 | 943 / 500 |

With all 4 A55 cores (`taskset -c 0-3`): rknn 203 / 1019 ms, rknn:int8 171 /
731 ms, rknn:mm 262 / 1202 ms (L1CA / E1B). (rknn rows: with the prefetch of
the output accumulation below; before: A76 166 / 916, A55 244 / 1266 (CPU 116
/ 542), A55 x4 207 / 1057 ms; rknn:int8 and other rows: before.)

- Full regression (309 signals, A76, default model `_f`): PASS, same
  detection for all signals, max diff C/N0 0.3 dB, code 0 sample, fine Doppler
  0.041 bin, peak 3.7% (same as the FP32/OpenVINO results of section 8.1);
  157 ms / CPU 18-21 ms per 1 ms-code search, 821-824 ms / CPU 59-67 ms per 4
  ms-code search. (`_w`: also PASS, 250 / 1050 ms.)
- Versus FFTW on the same core (default `_f`): A76 1.5x (L1CA) / 1.5x (E1B)
  faster with 1/8-1/9 of the CPU time; A55 4.6x / 4.4x faster with 1/10 of the
  CPU time. On the A55 the host steps (output accumulation into P, 0.65 ms per
  chunk and call) are now a large part of the search time. The accumulation
  (FP16 output in the native layout -> transpose -> P += scale * out, 8 rows
  of P at a time) stalls the in-order A55 at the cache misses of the
  read-modify-write of P (E1B: 6.9 MB of P per chunk > L3 3 MB, 0.9 GB/s).
  A software prefetch of the next cache line of each P row and of the output
  halves it on the A55 (E1B 18.9 -> 9.1 ms per chunk and call, 1.9 GB/s;
  next line: best, 2-16 lines ahead: 9.7-14.9 ms, slower with the distance)
  and does not change the A76 (2.6 -> 2.8 ms). The NPU search time is almost independent of the CPU core,
  so the NPU makes the acquisition on the low-power A55 cores as fast as on
  the A76.
- Per call (one non-coherent step, all bins, random data, A76): default model
  `_f` 7.6 ms / CPU 0.79 ms (L1CA, 21 bins), 161 ms / CPU 8.6 ms (E1B, 81
  bins); `_fq` (INT8) 5.0 / 0.82 ms, 90 / 8.8 ms; `_w` 12.3 ms / CPU 0.93 ms,
  207 ms / CPU 9.8 ms;
  host wipe-off model 10.4 / 1.58 ms, 180 / 21.6 ms; matmul API 7.0 / 6.6 ms,
  122 / 94 ms. The remaining CPU of the default model is mainly the
  accumulation of the output powers into P (E1B 2.4 ms per chunk, memory
  bound).
- `test_sdr_acc` (`SDR_ACC_TEST=rknn,rknn:hostwipe,rknn:mm`): all modes
  within the tolerances (NRMSE 0.0146 vs the CPU reference, carrier LUT of the
  reference).

### 13.5 INT8

First attempt (model `_w`, whole model quantized by RKNN-Toolkit2 w8a8 with
random calibration data): NPU 6.3 -> 3.9 ms, but NRMSE of the powers 8-11%
(the toolkit re-quantizes the output of every op, ~10 quantization points),
also with the IDFT in FP16; `w16a16i` is not faster than FP16.

Model `_fq` (`gen_rknn_acq.py -q`, option `int8`), INT8 of the folded model
`_f` with:

- hybrid quantization: the last IDFT stage F_A^-1 and the correlation power
  in FP16 (an INT8 power output quantizes the noise floor to a few levels and
  the peaks of strong signals were clipped: L5I C/N0 45.9 dB-Hz -12% peak),
- calibration data of synthetic signals of the real codes of the shape (64
  samples, codes of the period M / 2 / fs, e.g. L1CA, B1I, G1CA, L5I, E5aI,
  E6B for 1 ms; E1B, L6D for 4 ms; C/N0 30-60 dB-Hz, 1/4 noise only, random
  code offset and Doppler), IF samples at unit RMS,
- code DFT normalized by its max magnitude (|code| <= 1, section 8.1), the
  scale restored to the output; INT8 inputs written by the host in the native
  layout (C2 = 16).

Results: NPU time -29% (L1CA) / -38% (E1B) vs `_f` (table of 13.3); search
time A76 121 / 584 ms (FFTW 252 / 1394). 30 signals: PASS (max diff C/N0 0.3
dB, code 1 sample, peak 3.1%). 309 signals: same detection for all signals,
1 error: L6D PRN 199 (C/N0 36.7) has two CSK peaks of almost equal power
(1.13e-3 / 1.16e-3) and the INT8 noise selects the other one (code phase
+407 samples). With only the power in FP16: 4 errors (C/N0 of undetected
noise maxima 0.6 dB, L5I peak 6%, the L6D peak). `test_sdr_acc`: NRMSE 0.048
(weak signal, PASS) and 0.127 (strong signal C/N0 55.7, tolerance 0.05 of the
FP backends, C/N0 diff 0.15 dB; `test_sdr_acc` uses the tolerance 0.2 for
`rknn:int8`).

Interference (synthetic L1CA, fs 4 MHz, 10 ms, AGC to the total RMS and
4-bit IF, interference at Doppler +0 / 250 / 1234.5 Hz; J/N = interference
power / noise power):

| interference | CPU C/N0 | `_fq` (INT8) C/N0, NRMSE | `_f` (FP16) C/N0, NRMSE |
|---|---|---|---|
| none | 50.2 | 50.2, 0.048 | 50.2, 0.015 |
| CW J/N 10 dB | 35.0-37.0 | 47.5-48.1 (+11-13 dB), 0.95 | -0.02-+0.06 dB, 0.04 |
| CW J/N 20-50 dB (not detected) | 32.0-34.3 | 36.0-41.5 (+3.5-9.5 dB, false peaks), 0.97 | -0.7-0 dB, 0.04 |
| pulsed CW (duty 10 %) J/N 10-20 dB | 36.7-45.0 | +2.7-6.0 dB, 0.51-0.77 | -0.06-+0.04 dB, 0.02 |

A narrowband (CW) or pulsed interference concentrates the energy into a few
elements after the DFT stages and the fixed INT8 scales calibrated with the
interference-free signals saturate: the noise floor is suppressed and the
C/N0 metric is overestimated (false detections above the threshold 34 dB-Hz).
A uniform gain change (AGC) is cancelled by the unit RMS normalization. A
calibration including the interference would need ~sqrt(A) larger scales
(3 bits for A = 80) and lose the accuracy without interference. INT8 is
therefore an option (`rknn:int8`), not the default.

### 13.6 Tracking (CPU)

The NPU tracking backend is not implemented for RKNN (the correlation is a
batched GEMV bound by the memory traffic, section 11.4). Batched correlators
on the CPU (`bench_sdr_corr`, L1CA 16 MHz, us per channel and epoch, 16 / 64
/ 256 channels):

| backend (cores) | 4 taps | 21 taps |
|-----------------|--------|---------|
| cpu (1 A76) | 21.2 / 22.4 / 29.0 | 66.4 / 88.9 / 92.0 |
| cpu_mt:4 (4 A76) | 6.0 / 5.6 / 7.4 | 15.2 / 26.7 / 28.3 |
| cpu_mt:8 (4 A76 + 4 A55) | 8.0 / 5.1 / 6.5 | 17.8 / 26.0 / 14.5 |
| cpu (1 A55) | 96.6 / 98.1 / 108 | 258 / 302 / 297 |
| cpu_mt:4 (4 A55) | 27.0 / 25.3 / 26.0 | 65.8 / 74.4 / 73.2 |

Channels in real time (1 ms batch, 4 taps): 1 A76 ~45, 4 A76 ~170, 8 cores
~150-190, 4 A55 ~38. `quick_trk.sh` (33 channels L1CA/E1B/B1I, 1 s): `cpu`
and `cpu_mt` bit-identical to the original channel update on all core sets.

### 13.7 End-to-end receiver (pocket_trk, first 10 s, 10 signal types, 309 channels, 8 cores)

| `-acc` / `-tacc` | wall (s) | user CPU (s) | sys (s) | SIGNAL FOUND / LOST |
|------------------|----------|--------------|---------|---------------------|
| cpu / - | 11.7 | 14.8 | 1.4 | 7 / 1 |
| rknn / - | 11.8 | 8.2 | 1.9 | 9 / 1 |
| cpu / cpu_mt | 11.8 | 14.8 | 1.9 | 7 / 0 |
| rknn / cpu_mt | 12.7 | 8.8 | 2.6 | 8 / 1 |

The receiver searches one channel at a time (`update_srch_ch()`), so within
10 s only L1CA was searched; the CPU time of the acquisition is reduced by
~6.6 s per 10 s (0.66 core). A longer data set is needed for the all-in-view
evaluation.

Core assignment (`pocket_trk -cpu_acq cpus -cpu_trk cpus`, `sdr_thread_affinity()`):
the threads of the `-acc` / `-tacc` backends inherit the CPU affinity of the
caller of `sdr_acc_init()` / `sdr_acc_trk_init()`; the channel threads switch
between the two CPU sets by the channel state (search: `-cpu_acq`, lock:
`-cpu_trk`), the batched correlator group threads and the IF input thread run
on `-cpu_trk`, and `cpu_mt` without `:n` uses as many threads as the CPUs of
the affinity. `rknn / cpu_mt` with `-cpu_acq 0-3 -cpu_trk 4-7` (A55 + NPU for
the acquisition, A76 for the tracking): 312 threads on 0-3 and 14 on 4-7,
same signals found (9), batched tracking 0.96 -> 0.50 s per 10 s (cpu_mt:4
on the A76 instead of cpu_mt:8 on all cores), acquisition 13.9 -> 13.0 ms per
call (NPU bound, not slower on the A55). Without `-tacc` the locked channel
threads move to 4-7.

### 13.8 End-to-end receiver on the full data (300 s)

Conditions: the full IF data set (19.2 GB, 300 s), read from a NAS (SMB 3.0,
1GbE, 118 MB/s) at the real-time rate 64 MB/s. The SD card of the ROCK 5B
(90 MB/s) was saturated by the 64 MB/s replay under a high CPU load (SD
utilization 90-97%, the IF input thread waiting for the block I/O), which
limited the first capacity runs at ~150 channels; with the NAS the input
thread does not wait for the I/O. Real time = the 300 s data processed in
~302-305 s of wall time (`pocket_trk` file input waits when ahead and falls
behind when the input thread cannot keep up). Fan cooling (max 60 C, no
throttling).

10 signal types of `acq_sigs.sh` (309 candidates, data components only):

| configuration | search cycle of 309 ch | LOCK at 300 s | sats used / valid | max BUFF | user CPU |
|---|---|---|---|---|---|
| `-acc cpu` (PocketFFT) | ~240 s | 66 | 33 / 54 | 27% | 17.1 min |
| `-acc rknn` | ~115 s | 73 | 33 / 55 | 11% | 18.2 min |
| `-acc rknn -cpu_acq 0-3 -cpu_trk 4-7` | ~115 s | 73 | 33 / 53 | 13% | 12.4 min |
| `-acc rknn -tacc cpu_mt` (+ affinity) | ~120 s | 72 | 17 / 23 | 93% | 11.2 min |
| `-acc cpu -tacc cpu_mt` | ~240 s | 52 | 12 / 24 | 95% | 14.5 min |

The NPU halves the search cycle; with the A55 for the acquisition (A55 15%,
A76 64-69%) the CPU time is -27%. The batched tracking correlators (`-tacc`)
do not scale to this size: one group thread per (RF channel, code period)
runs the non-correlator parts of all its channels serially (`sdr_ch_update_pre`
/ `post`), the IF buffer usage reaches 93-95% and the PVT loses satellites.
On the CPU the batch has no benefit per channel (Raspberry Pi 5: 30.5 vs 30.7
us per channel-epoch), so the channel threads are the better CPU design.

### 13.9 All-in-view capacity

All-in-view signal set: the visible satellites of the data (GPS 9, Galileo
7, BeiDou 19, GLONASS 6, QZSS 4) x all supported signals incl. the pilots in
the 8 RF channels (L1CA, L1CD/P, E1B/C, B1CD/P, B1I, G1CA, L2CM, G2CA, L5I/Q,
E5aI/Q, B2aD/P, E5bI/Q, B2I, B2bI, L6D, E6B/C; 232 candidates): 185 signals
found, 176 locked at the end (`cand_sigs.sh`). L6D/L6E are excluded from the
capacity runs (the CSK demodulation by a chip-domain FFT per code period
costs 0.23 core per channel, 6x of the other signals).

SBAS, IRNSS (NavIC) and QZSS augmentation signals (`cand2_sigs.sh`, 174
candidates, 300 s, rknn): 33 signals found, ~30 locked at the end: SBAS L1CA
PRN 122, 126, 128, 129, 130, 134, 137, 142, 143, SBAS L5I 122, 128, 130,
134, 142, 143, SBAS L5Q 130, 134, 142, 143 (134 and 143 at C/N0 15-29 dB-Hz
at the end: possible false locks, the SBAS L5Q secondary code to be
checked), IRNSS I5S 3, 7, 10, I1SD/I1SP 10, QZSS L1S 184, 185, 189 and
L5SI/L5SQ 184, 185, 189. The all-in-view set of the data is therefore ~200
signals without L6 (~206 with L6D).

Capacity runs: N tracked signals (stratified by signal type from the 172
locked signals except L6D; N > 172 by duplicated channels) + 227 never
detected candidates of `acq_sigs.sh` searched continuously (one search at a
time), `capx_N.sh`. The CPU-only reference is FFTW with wisdom of all DFT
sizes (`USE_FFTW=1`, PATIENT/MEASURE wisdom of 32000, 128000, 320000,
640000; the FFT of the tracking is only the L6 CSK).

| N | configuration | wall (s) | LOCK at end | CPU (A55 / A76) | user CPU | max BUFF |
|---|---|---|---|---|---|---|
| 125 | FFTW | 304 | 118 | 75% | 27.1 min | 23% |
| 125 | rknn | 304 | 113 | 67% | 23.6 min | 12% |
| 125 | rknn, acq on A55 | 304 | 118 | 67% | 23.1 min | 13% |
| 150 | FFTW | 304 | 139 | 83% | 29.5 min | 29% |
| 150 | rknn | 305 | 138 | 78% | 27.3 min | 11% |
| 150 | rknn, acq on A55 | 305 | 140 | 77% | 26.2 min | 14% |
| 172 | FFTW | 304 | 162 | 88% (89 / 87) | 31.0 min | 28% |
| 172 | rknn | 305 | 164 | 88% (90 / 87) | 30.4 min | 13% |
| 172 | rknn, acq on A55 | 305 | 160 | 85% (88 / 83) | 28.6 min | 11% |
| 200 | FFTW | 304 | 167 | 93% | 32.4 min | 46% |
| 200 | rknn | 312 | 187 | 95% | 32.7 min | 18% |
| 200 | rknn, acq on A55 | 305 | 186 | 90% | 30.3 min | 23% |
| 230 | FFTW | 318 | 189 | 95% | 34.4 min | 66% |
| 230 | rknn | 335 | 220 | 93% | 36.0 min | 12% |
| 230 | rknn, acq on A55 | 316 | 191 | 93% | 32.2 min | 32% |
| 260 | FFTW | 337 | 211 | 95% | 36.9 min | 72% |
| 260 | rknn | 347 | 237 | 94% | 37.6 min | 19% |
| 260 | rknn, acq on A55 | 327 | 209 | 93% | 33.4 min | 18% |

(rknn: `-acc rknn`; acq on A55: `-cpu_acq 0-3 -cpu_trk 0-7`, the tracking on
all cores. These runs used the matmul API for the assisted searches, see
13.2.)

- The all-in-view set without L6 (172 signals, 3 frequencies x 5
  constellations incl. pilots) is tracked in real time by all
  configurations while 227 other signals are searched continuously; the
  limit is ~200 channels (CPU 90-95%).
- At N=200 the NPU with the acquisition on the A55 keeps 186 channels locked
  in real time vs 167 by FFTW (+11%). The difference is the acquisition
  throughput under load: with the CPU saturated the FFTW search slows down
  and the assisted searches of the long codes (B1CP 7, E1C 3, L1CP 2, L2CM,
  found by the NPU at 176-193 s in the second search cycle) are not reached
  within 300 s; the IF buffer usage of FFTW goes up to 46-72% (NPU 12-32%).
- Without the core assignment the host steps of the NPU search compete with
  the tracking on the A76 (N=200: 312 s, behind real time); with the
  acquisition on the A55 the receiver stays in real time.
- The NPU saves only 0.4-0.7 core at the same load: the receiver searches one
  signal at a time, so the CPU acquisition uses at most ~1 core of 8, and the
  tracking (~0.05 core per channel) is the limit of the capacity.

### 13.10 Direct acquisition of long codes

`test/acc/long_sigs.sh` (11 GPS PRNs, L1CP 10 ms: M=320000, 201 bins; L2CM
20 ms: M=640000, 401 bins), single core, time / CPU time per signal, same
detections for all (5 detected, max diff C/N0 0.1 dB, code 0, Doppler 0.006
bin):

| backend | L1CP A76 | L2CM A76 | L1CP A55 | L2CM A55 |
|---|---|---|---|---|
| FFTW (wisdom) | 5.46 s / 5.44 s | 11.2 s / 11.2 s | 25.7 s / 25.7 s | 55.2 s / 55.1 s |
| rknn (`_f`, chunks 23 / 20 bins) | **2.97 s / 0.27 s** | **5.72 s / 0.65 s** | **3.31 s / 0.69 s** | **6.54 s / 1.54 s** |
| rknn:mm | 2.17 s / 1.44 s | 5.98 s / 2.98 s | 6.91 s / 6.15 s | 15.8 s / 12.7 s |

The NPU is 1.8-2.0x faster than FFTW on the A76 with 1/17-1/20 of the CPU
time, and 7.8-8.4x faster on the A55 with 1/36 of the CPU time (almost the
same search time as from the A76). The large FFTs are slow on the RK3588 CPU
(L1CP 5.5 s on the A76 vs 0.61 s on an Intel P-core, section 8.0).

### 13.11 Pull-in failures (SIGNAL LOST)

The NPU runs had more SIGNAL LOST (18-37 vs 21-24 per run). All LOSTs are
signals of the tracked set (no false alarm of the background), ~2 s after
the acquisition (pull-in failure), at C/N0 < 38 dB-Hz at the acquisition.
With the same IF data and start time (bench_sdr_trk, 66 weak signals, 14
start times, acquisition by cpu or rknn) the failures are the same (48 vs 49
of 807), so the NPU acquisition has the same quality; the receiver with the
NPU re-acquires the weak signals more often (faster search cycle). The
failures are mostly the 10.23 / 5.115 Mcps signals of the L5 band (E5aI 62%,
E5aQ 58%, B2aD 45%, L5I 33%, E6 24-31%; <= 2.046 Mcps: 0-2%), weak data
components found by the assisted acquisition; a wider E-L spacing (0.5, 1.0
chip) does not reduce them (bench: 10-12%), so they are a property of the
tracking loops, not of the 16 MHz sampling (the signal power of BPSK(10) in
+/-8 MHz is 89.4% vs 90.6% in +/-12 MHz). The 24 MHz sampling is still
needed for the code tracking accuracy of the wideband signals.

### 13.12 Raspberry Pi 5 (CPU only)

Cortex-A76 x4 2.4 GHz, 8 GB, Ubuntu 24.04, same code and data (USB3 SSD):

| | Pi 5 | ROCK 5B A76 |
|---|---|---|
| L1CA search PocketFFT / FFTW | 396 / 246 ms | 437 / 252 ms |
| E1B search PocketFFT / FFTW | 3329 / 2053 ms | 2094 / 1394 ms |
| correlator cpu (1 core), 16 / 64 / 256 ch | 20.2 / 20.7 / 26.8 us | 21.2 / 22.4 / 29.0 us |
| correlator cpu_mt:4, 16 / 64 / 256 ch | 6.2 / 5.8 / 6.9 us | 6.0 / 5.6 / 7.4 us |

Same A76 performance except the large FFT (128000 points: 1.5x slower, 2 MB
L3 and narrower memory bus). The ROCK 5B adds the 4 A55 and the NPU. The E2E
capacity runs on the Pi 5 stopped (the board shut down under the full load
with a 5V 3A supply and the USB SSD; a 5V 5A supply is needed).

### 13.13 Shared DFT on the NPU (shift mode)

With the bins spaced by fs / M, the correlation of bin k is
|IDFT(X[m+k] C[m])| = |IDFT(X[m] C[m-k])|, where X is the DFT of the IF
samples wiped by the carrier of the first bin of the chunk. The model `_s`
computes the forward DFT (F_A, twiddle, Transpose, F_B) once, replicates it for
the bins and multiplies it by the code DFT rolled by k bins (an input written
by the host when the code changes, 16-byte copies in the native layout from
rows extended by the wrapped element), followed by the same IDFT as `_f`. The
replication by `Tile` (bins along W) or by a broadcast `Mul` (bins in the
batch) gave wrong results on the RK3588 NPU (NRMSE 0.83-0.90 for all bins,
checked with RKNN-Toolkit-Lite2 on the board) although correct in the
RKNN-Toolkit2 simulator (1.4e-3); with `Concat` of Kc copies the NPU result
is correct (1.4e-3, same as `_f`).

`quick_acq.sh` (30 signals, PASS, same as `_f`), time / CPU time per search:

| | L1CA A76 | E1B A76 | L1CA A55 | E1B A55 |
|---|---|---|---|---|
| `_f` | 168 / 32 ms | 914 / 165 ms | 232 / 103 ms | 1145 / 423 ms |
| `_s` (default) | **119 / 29 ms** | **576 / 148 ms** | **177 / 97 ms** | **811 / 392 ms** |

Long codes (A76, 2 PRNs): L1CP 3.36 -> 1.83 s, L2CM 6.42 -> 3.72 s per
search (FFTW 5.46 / 11.2 s). The model sizes are also smaller (no folded
weights per bin, e.g. E1B 128000:27 7.3 MB).

### 13.14 Live-like capacity, CPU profile and decimation

CPU profile (`perf record -a`, 30 s, N=200 of 13.9): `rknn` with the
acquisition on the A55: the tracking correlator 74% of all CPU time
(`mix_carr_p` carrier mixing by LUT 42%, `dot_IQ_code_acc` 32%), IF input
thread 4.4%, libc 7.7%, kernel 8.6%, the NPU plugin 1.2% (~0.1 core);
FFTW: `libfftw3f` 7.4% (~0.6 core) plus the host steps of the CPU search.
With the Kc = 1 models of the assisted searches (13.2) the capacity did not
change (N=200: 183 vs 186 channels), so the NPU path costs ~0.1 core and the
capacity is set by the tracking correlators.

Live-like input: with the file input the IF input thread waits when the CPU
is saturated and the whole receiver falls behind (13.9). With
`pocket_trk -rcv_prio 10` (SCHED_RR, `setcap cap_sys_nice+ep`) the input
stays in real time like a live front-end, and an overload shows as the
IF buffer usage > 100% (the channels lag behind the buffer, data lost). The
capacity is then the largest N without overflow. Decimation (`-dec
1:2,2:2,3:2,4:2,5:2`: the RF CHs of L1/E1/B1C, B1I, G1, L2C, G2 at 8 MHz by
the 9-tap LPF and 2:1 decimation, tracking and search at 8 MHz; L5/E5/E6 at
16 MHz; 8 MHz models 16000:7:63, 16000:1:63, 64000:27, 64000:1, 160000:1 and
FFTW wisdom of the same sizes; C/N0 of the decimated signals -0.5 dB):

| N | configuration | LOCK at end (time to reach) | max BUFF | CPU | user CPU |
|---|---|---|---|---|---|
| 200 | rknn, acq on A55 | 184 | 85% | 97% | 31.5 min |
| 200 | FFTW | 170 | 52% | 93% | 32.5 min |
| 230 | rknn, acq on A55 | 182 | 125% (overflow) | 100% | 32.5 min |
| 230 | FFTW | 190 | 188% (overflow) | 101% | 33.2 min |
| 200 | rknn, acq on A55, decimation | 191 | 7% | 84% | 29.5 min |
| 200 | FFTW, decimation | 193 | 16% | 88% | 32.1 min |
| 230 | rknn, acq on A55, decimation | **217** (181 s) | 10% | 91% | 31.6 min |
| 230 | FFTW, decimation | 214 (242 s) | 25% | 92% | 33.5 min |
| 260 | rknn, acq on A55, decimation | 230 (max 238) | 119% (overflow) | 99% | 32.6 min |
| 260 | FFTW, decimation | 214 (search cycle not completed in 300 s) | 26% | 94% | 34.0 min |

(rknn: shared DFT models `_s`; all with `-rcv_prio 10`; single runs.)

Comparison with the CPU backends with the decimation (live-like):

| N | configuration | LOCK at end | max BUFF | CPU | user CPU |
|---|---|---|---|---|---|
| 230 | PocketFFT `cpu` (original default) | 207 | 21% | 90% | 31.4 min |
| 230 | FFTW `cpu` | 214 | 25% | 92% | 33.5 min |
| 230 | FFTW `cpu_opt` (shared DFT on the CPU) | 218 | 17% | 93% | 33.9 min |
| 230 | rknn `_s`, acq on A55 | 217 | 10% | 91% | 31.6 min |
| 260 | PocketFFT `cpu` | 210 | 47% | 94% | 32.3 min |
| 260 | FFTW `cpu` | 214 | 26% | 94% | 34.0 min |
| 260 | FFTW `cpu_opt` | 235 (max 237) | 71% | 99% | 34.6 min |
| 260 | rknn `_s`, acq on A55 | 230 (max 238) | 119% (overflow) | 99% | 32.6 min |

Acquisition with the shared DFT on both sides (`quick_acq.sh`, PASS), time /
CPU time per search: A76 FFTW `cpu_opt` L1CA 129 / 129 ms, E1B 625 / 609 ms
vs rknn `_s` 119 / 29 ms, 585 / 138 ms; A55 563 / 563 ms, 3692 / 3676 ms vs
176 / 97 ms, 795 / 383 ms.

- With the shared DFT on the CPU (`cpu_opt`) the best CPU-only receiver tracks
  as many channels as the NPU receiver (N=230: 218 vs 217; N=260: 235 vs
  230-238, single runs). The serial search uses at most ~1 core, and with
  `cpu_opt` that core is used more efficiently, so the NPU does not increase
  the channel capacity of this receiver on the RK3588; it reduces the CPU
  time of the search to 1/4-1/10 (A55: 3-5x faster search) and the CPU time
  of the receiver by 0.4-0.6 core at the same load. The capacity gains come
  from the decimation and the shared DFT, i.e. from the tracking cost and the
  search efficiency, not from the offload.

- Without decimation the real-time limit is ~185 channels (N=200: NPU 184
  locked at BUFF 85%, FFTW 170; N=230: both overflow).
- The decimation of the narrowband RF CHs raises it to ~217-230 channels
  (+17-24%) with the same CPU, independent of the NPU.
- The NPU reaches the locked set earlier (N=230: 217 at 181 s vs 214 at 242
  s by FFTW), uses ~0.5 core less at the same load, and near the limit keeps
  acquiring up to the tracking capacity (N=260: 230-238, then overflow),
  while the FFTW search does not complete the cycle of 487 channels in 300 s.

### 13.15 NEON carrier mixing of the tracking correlator

The carrier mixing (`mix_carr_p()`, 42% of the CPU time in 13.14) looked up
a LUT of 64K entries (IF sample x 256 carrier phases) one sample at a time.
The NEON version looks up the carrier (I, Q) of 16 samples by one 64-byte
table lookup (`vqtbl4q`, 64 phases = 6 bits) and multiplies in int8 (the
wrapped products give the exact sum, |result| <= 127). With 256 phases (4
chained lookups) it was slower than the LUT (A76 22.8 vs 20.9 us). The phase
error of 6 bits is +/-2.8 deg (correlation loss 0.002 dB). `-DMIX_LUT_SCALAR`
builds the original LUT.

- Correlator (`bench_sdr_corr`, L1CA, 4 taps, us per channel-epoch, 16 / 64 /
  256 ch): A76 20.9 / 21.0 / 22.9 -> 14.4 / 14.6 / 16.2 (1.45x), A55 95.0 /
  94.3 / 94.0 -> 52.6 / 52.4 / 55.4 (1.8x). Whole tracking (`bench_sdr_trk`,
  50 ch, 5 s, A76): 30.1 -> 23.1 us per channel-epoch (1.30x).
- Accuracy vs the LUT (same data): C/N0 within +/-0.02 dB, code < 0.1 m,
  Doppler 0.01-0.05 Hz and no ADR difference for all channels except 3 weak
  ones (C/N0 34-35 dB-Hz) whose loops took different paths. Acquisition
  regression (`quick_acq.sh`, cpu, fftw+cpu): PASS.

x86 (AVX2, Intel Core Ultra; evaluation code in commit 9c1e219, removed
afterwards, `-DMIX_PHASE6_AVX2`): the 64-entry lookup needs
4 `vpshufb` + 3 blends and the int8 products `vpmaddubsw` on |data| with the
sign moved to the carrier; correlator (16 / 64 / 256 ch) P-core 6.1 -> 5.55 us
(1.10x), E-core 12.2 -> 14.3 us (0.85x), LP E-core 13.1 -> 15.3 us (0.86x);
the tracking results are the same as the NEON version (`cmp_trk.py` PASS, lock
and sync states identical). The x86 build therefore keeps the LUT: the
improvement is for NEON (ARM) only.

Capacity with decimation, live-like (13.14 conditions):

| N | configuration | LOCK at end | max BUFF | CPU (A55 / A76) | user CPU |
|---|---|---|---|---|---|
| 260 | rknn `_s`, acq on A55 | 252 | 6% | 81% (85 / 77) | 27.5 min |
| 260 | FFTW `cpu_opt` | 249 | 8% | 87% (89 / 86) | 31.4 min |
| 300 | rknn `_s`, acq on A55 | 284 | 8% | 88% (91 / 86) | 29.5 min |
| 300 | FFTW `cpu_opt` | 286 | 13% | 92% (92 / 91) | 32.6 min |

The capacity rises from ~217-230 to > 285 channels (the limit is estimated at
~320). The NPU receiver uses ~0.6 core less CPU at the same load.

### 13.16 Findings

- On RK3588 the NPU acquisition (default `_f`) is 1.5x faster than FFTW on
  the A76 and 4.8-5.0x on the A55 with 1/8-1/14 of the CPU time (INT8: 2.1-2.4x
  on the A76, an option because of the saturation by interference); for the
  direct search of the long codes 2x (A76) and 8x (A55) with 1/17-1/36 of
  the CPU time. The model-level optimizations (carrier folding, batch-1
  layout, 3 real Conv) reduced the NPU time by 38% (L1CA) / 22% (E1B); the
  remaining NPU time is Conv, Transposes and the element-wise twiddle/code
  products.
- In the receiver the NPU halves the search cycle of all signals and, with
  the acquisition on the A55, keeps more channels locked in real time near
  the CPU limit than FFTW without the shared DFT (N=200: 186 vs 167), because
  its search does not slow down when the CPU is saturated. Against the best
  CPU-only receiver (FFTW `cpu_opt`, shared DFT) the channel capacity is the
  same (13.14); the NPU saves 0.4-0.6 core and makes the search on the A55 as
  fast as on the A76.
- ROCK 5B tracks the all-in-view set without L6 (172 signals) in real time
  with continuous search of 227 other signals; with a live-like input the
  limit is ~185 channels, ~217-230 with the decimation of the narrowband RF
  CHs to 8 MHz (13.14) and > 285 with the NEON carrier mixing (13.15). The capacity is set by the tracking correlators
  (74% of the CPU); the NPU acquisition costs ~0.1 core (FFTW ~0.6 core).
- The host CPU cost of the NPU path is the I/O in the native layout; FP32 or
  NHWC inputs cost more CPU than the FFT itself.
- The matmul API (no precompiled model) is the fastest wall time on the A76
  but uses as much CPU as the FFT (host element-wise steps, memory bound).
- The shared DFT (`_s`, default) reduces the NPU search time to 0.55-0.76x of
  `_f` (13.13): vs FFTW on the A76 1.4x (L1CA) / 2.4x (E1B) / 3x (long codes)
  faster.
- Open: 24 MHz data and a live Pocket SDR FE, ROCK 5C.
