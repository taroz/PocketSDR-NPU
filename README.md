# **PocketSDR-NPU - GNSS Signal Acquisition on NPUs of Single-Board Computers**

**PocketSDR-NPU** runs the GNSS signal acquisition (parallel code search) of
[**Pocket SDR**](https://github.com/tomojitakasu/PocketSDR) on the NPU of a
low-power single-board computer (SBC), e.g. Raspberry Pi 5 with AX8850
(M5Stack LLM-8850, PCIe) or an RK3588 board (built-in NPU). The NPU makes the
acquisition faster than the CPU (FFTW), 3.0-5.3 times on the Raspberry Pi 5
with AX8850 and up to 8.6 times on the Cortex-A55 of RK3588, and reduces its
CPU time, so the receiver searches more signals and locks them faster. On the
Raspberry Pi 5 with AX8850, the receiver tracks about 230 signals in real time
instead of about 175 with the acquisition on the CPU. The long codes (L1CP
10 ms, L2CM 20 ms) can be acquired directly over the full Doppler range.

The receiver itself (tracking, navigation data, PVT) is Pocket SDR. For Pocket
SDR, its RF front-ends and its documents, see the
[**original repository**](https://github.com/tomojitakasu/PocketSDR).

---

## **How It Works**

- The parallel code search is a DFT-based circular correlation of the IF data
  and the code for each Doppler bin. With the Doppler bins spaced by fs / M
  (M: DFT size), all bins share one DFT of the IF data shifted by the bin.
- The DFT is computed on the NPU as a four-step DFT (matrix products, twiddle
  factors and transposes). One inference processes a chunk of Doppler bins.
- The division of the work depends on the connection of the NPU. On the
  RK3588 (shared memory), the whole search including the DFT of the IF data
  runs on the NPU (INT8, the last IDFT and the power in FP16). On the AX8850
  (PCIe), the host computes the DFT of the IF data once (FFTW), and the NPU
  computes the correlation in 16-bit, integrates the windows and finds the
  peak on the card, so only the peaks are copied back to the host.
- The acquisition is a plugin (`libsdr_acc_<name>.so`) loaded at runtime by
  `-acc <name>`. Signals without a model are searched by the CPU.

## **Results**

Acquisition time per signal (all PRNs, mean of 5 runs), 24 MHz IF data of
Pocket SDR FE 4CH (`pocket_L1L2L5E5b_24MHz.conf`), the narrowband signals
decimated to 8 MHz. CPU: the same search on the CPU with FFTW. The results of
the NPUs are in bold.

Raspberry Pi 5 (Cortex-A76 2.4 GHz x 4), search time per signal (ms) and
speed-up to the CPU (x):

| Method | L1CA (8 MHz) | B1I (8 MHz) | E1B (8 MHz) | L5I (24 MHz) | L1CP (8 MHz, direct) | L2CM (8 MHz, direct) |
|---|---|---|---|---|---|---|
| CPU | 51 | 51 | 275 | 197 | 1088 | 2936 |
| AX8850 (M5Stack LLM-8850) | **15** (3.4x) | **15** (3.4x) | **72** (3.8x) | **66** (3.0x) | **206** (5.3x) | **581** (5.1x) |
| Hailo-8 (AI HAT+) | **23** (2.2x) | **23** (2.2x) | **88** (3.1x) | **72** (2.7x) | **411** (2.6x) | - (*1) |
| Hailo-10H (AI HAT+ 2) (*2) | **64** (0.8x) | **62** (0.8x) | **134** (2.1x) | **113** (1.7x) | **646** (1.7x) | **1127** (2.6x) |

(\*1) No model (too large for the Hailo-8), searched by the CPU.
(\*2) Mean of 2 runs.

ROCK 5C (RK3588, Cortex-A76 2.4 GHz x 4 + Cortex-A55 1.8 GHz x 4, the search
on the A76 or the A55), search time per signal (ms) and speed-up to the CPU on
the same cores (x):

| Method | L1CA (8 MHz) | B1I (8 MHz) | E1B (8 MHz) | L5I (24 MHz) | L1CP (8 MHz, direct) | L2CM (8 MHz, direct) |
|---|---|---|---|---|---|---|
| CPU (A76) | 52 | 52 | 306 | 203 | 1123 | 2918 |
| RK3588 NPU (A76) | **45** (1.1x) | **45** (1.2x) | **165** (1.9x) | **132** (1.5x) | **513** (2.2x) | **1231** (2.4x) |
| CPU (A55) | 263 | 259 | 1536 | 1002 | 5557 | 13213 |
| RK3588 NPU (A55) | **65** (4.0x) | **65** (4.0x) | **207** (7.4x) | **181** (5.5x) | **645** (8.6x) | **1584** (8.3x) |

- The detections are the same as the CPU. The sensitivity of the AX8850 and
  the RK3588 NPU (C/N0 at the
  probability of detection 0.5) differs from the CPU by 0.06 dB or less
  (0.14 dB or less for L1CP).
- The CPU time of the acquisition is 1/5.6-1/26 of the CPU on the Raspberry
  Pi 5 with AX8850 and 1/3.7-1/13 on the RK3588.
- In the receiver (300 s, fixed sets of 140-260 signals), the NPU locks
  1.1-1.5 times more signals by 30 s with the search on the Cortex-A76 and
  1.9 times with the search on the Cortex-A55. On the Raspberry Pi 5 with
  AX8850, the real-time limit of the tracked signals increases from about 175 to about 230.

## **Supported NPUs**

| NPU | Plugin | Runtime | Model generator |
|---|---|---|---|
| Axera AX8850 (M5Stack LLM-8850, M.2 PCIe) | `libsdr_acc_axcl.so` | AXCL 3.6 | `lib/build/gen_axcl_acq.py` (Pulsar2 8.0) |
| Hailo-8 (Raspberry Pi AI HAT+) | `libsdr_acc_hailo.so` | HailoRT 4.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Hailo-10H (Raspberry Pi AI HAT+ 2) | `libsdr_acc_hailo.so` | HailoRT 5.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Rockchip RK3588 NPU | `libsdr_acc_rknn.so` | RKNN runtime 2.3 | `lib/build/gen_rknn_acq.py` (RKNN-Toolkit2) |

The models depend on the NPU, the runtime version and the sampling rate. The
models for the 24 MHz configuration above are in the release assets.

| Model directory | Models (`acq_M<M>_K<Kc>_R<R>`) |
|---|---|
| `lib/linux/axcl` | AX8850 axmodels (`_hs`: DFT of the IF data by the host, `_w<n>`: n windows, `_red`: peak search on the card) |
| `lib/linux/hailo8` | Hailo-8 HEFs (no L2CM, too large for the Hailo-8) |
| `lib/linux/hailo` | Hailo-10H HEFs |
| `lib/linux/rknn` | RK3588 INT8 RKNN models |

## **Build**

Requirements: Linux (ARM64), FFTW3 (`libfftw3-dev`), and the runtime of the
NPU (AXCL, HailoRT or the RKNN runtime).

```
cd lib/build
make USE_SOAPY=0 USE_FFTW=1 USE_AXCL=1         # AX8850 (USE_RKNN=1 for RK3588, USE_HAILO=1 for Hailo)
make USE_SOAPY=0 USE_FFTW=1 USE_AXCL=1 install
cd ../../app/pocket_trk
make USE_SOAPY=0 USE_FFTW=1 && make USE_SOAPY=0 USE_FFTW=1 install
```

## **Usage**

```
# Raspberry Pi 5 + AX8850
pocket_trk -acc axcl -dec 1:3:7.42,1:3:-6.902,2:3 -trk_guard 20 -rcv_prio 10 \
    -c pocket_L1L2L5E5b_24MHz.conf [-sig ... -prn ...] [file]

# RK3588
pocket_trk -acc rknn -dec 1:3:7.42,1:3:-6.902,2:3 -trk_guard 20 -rcv_prio 10 ...

# Raspberry Pi 5 + Hailo-8
pocket_trk -acc hailo:hef=../lib/linux/hailo8,batch=16 -dec 1:3:7.42,1:3:-6.902,2:3 ...
```

| Option | Description |
|---|---|
| `-acc backend` | Acquisition backend: `cpu` (default), `axcl[:model=dir]`, `rknn[:cores=n,model=dir]`, `hailo[:hef=dir,batch=n]` |
| `-dec rfch:D[:foff],...` | Decimation of the RF CH by D (e.g. 24 MHz to 8 MHz for the narrowband signals) |
| `-trk_guard pct` | No new search while the IF buffer of the tracking exceeds pct (%) |
| `-rcv_prio prio` | Real-time priority of the IF input thread (needs CAP_SYS_NICE) |
| `-cpu_acq cpus`, `-cpu_trk cpus` | CPU cores for the acquisition and the tracking |

`pocket_acq` also has `-acc`. See [doc/command_ref.md](doc/command_ref.md).

## **Evaluation and Tests**

- `test/eval`: acquisition benchmark (`acq_bench.sh`), fixed-set capacity of
  the receiver (`fixed_cap.sh`, `cap_sum.py`), acquisition sensitivity with
  synthetic IF data (`gen_sens.py`, `sens_test.sh`, `sens_sum.py`)
- `test/utest`: `bench_sdr_acq`, `bench_sdr_trk`, `test_sdr_acc`

## **Changes from Pocket SDR**

- Acquisition plugin interface (`src/sdr_acc.c`, `src/sdr_acc.h`, API version
  2 with the optional peak search `search_code_peak`) and the plugins for
  AX8850, RK3588 and Hailo (`src/sdr_acc_axcl.cpp`, `src/sdr_acc_rknn.cpp`,
  `src/sdr_acc_hailo.cpp`)
- Shared DFT of the IF data among the Doppler bins in `sdr_search_code()`,
  NEON carrier mixing of the tracking correlator, the fix of the 16 KB block
  order of FE 4CH/8CH after a USB reset (proposed to Pocket SDR)
- Receiver options for SBCs (`-dec`, `-trk_guard`, `-rcv_prio`, `-cpu_acq`,
  `-cpu_trk`)

Branches: `main` (this version), `dev` (development history with the other
evaluated methods).

## **License**

BSD 2-clause license, the same as Pocket SDR (see [LICENSE.txt](LICENSE.txt)).
Pocket SDR is developed by T. Takasu.
