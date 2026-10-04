# NPU Acceleration Architecture (Development Note)

Date: 2026-10-02

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
- Only the Intel NPU via OpenVINO is implemented. Static shapes; the first
  compilation of a model takes ~0.5 s (cached afterwards).
- The CPU usage of the NPU backend includes the busy polling of the NPU
  driver while waiting for the completion.
- Tracking (Phase 2): the NPU tracking backend is functional and verified
  but slower than the CPU on this platform (section 11), so `pocket_trk -tacc
  npu` cannot run in real time. Batched correlators exclude L6D/L6E and E5ABQ.
- The measurements were done on an x86 laptop SoC; the LP E-cores are used as
  a proxy of a low-power CPU. The target ARM SBCs need their own backends and
  measurements.

## 10. Supported NPU Backends (Acquisition and Tracking)

| backend | plugin | device | status |
|---------|--------|--------|--------|
| Intel NPU (Core Ultra, NPU 5) | `libsdr_acc_ov.so` (OpenVINO 2026.4) | `NPU` | implemented, verified |
| OpenVINO CPU (FP32, reference of the graph) | `libsdr_acc_ov.so` | `CPU` | implemented (test only) |
| Rockchip RK3588, AMD XDNA, Apple ANE, Qualcomm Hexagon | - | - | future (same plugin interface, own low-level implementation) |

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
