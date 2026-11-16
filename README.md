# **NPU-Accelerated GNSS Signal Acquisition**

**PocketSDR-NPU** uses the NPU (Neural Processing Unit) of a small single-board computer (SBC), which is
made for machine learning, as an accelerator for GNSS signal acquisition. It
is based on [**Pocket SDR**](https://github.com/tomojitakasu/PocketSDR), an open-source
software GNSS receiver, with the parallel code search offloaded to the NPU.

- The GNSS signal acquisition is **3.0-3.8 times faster** than an optimized CPU FFT.
- The long codes (L1CP 10 ms, L2CM 20 ms) can be acquired **directly over the
  full Doppler range** in a practical time (5.1-5.3 times faster).
- In the receiver, **1.5 times more signals are locked within 30 s**.
- The CPU is left to the tracking: the receiver **tracks about 230 signals in
  real time instead of about 175**.

The receiver itself (tracking, navigation data, PVT) is Pocket SDR. For Pocket
SDR, its RF front-ends and its documents, see the
[**original repository**](https://github.com/tomojitakasu/PocketSDR).

![PocketSDR-NPU](https://github.com/user-attachments/assets/41a2966c-8466-42da-b5d9-30a82f791a66)
---

## **How It Works**

- The DFT of the parallel code search is decomposed by the four-step DFT into
  matrix products of small DFTs, which an NPU computes efficiently
  (1x1 convolutions, element-wise products and transposes).
- The work is divided between the CPU and the NPU by how the NPU is connected:
  on the RK3588 (memory shared with the CPU), the whole search runs on the NPU;
  on the AX8850 (PCIe, its own memory), the CPU computes the DFT of the IF data
  once and the NPU does the rest including the peak search, so only the peaks
  are copied back.
- The acquisition is a plugin of Pocket SDR selected by `-acc <name>`.

## **Results**

Acquisition time per signal, 24 MHz IF data of
Pocket SDR FE 4CH (`pocket_L1L2L5E5b_24MHz.conf`), the narrowband signals
decimated to 8 MHz. CPU (FFTW): the same search on the CPU with FFTW. 

![Acquisition time on Raspberry Pi 5](https://github.com/user-attachments/assets/5aa2c583-f38c-4f43-9c3f-57a301412c82)

### Raspberry Pi 5 + NPU
- CPU: Cortex-A76, search time per signal (ms) and speed-up to the CPU (x):

| Method | L1CA (8 MHz) | B1I (8 MHz) | E1B (8 MHz) | L5I (24 MHz) | L1CP (8 MHz) | L2CM (8 MHz) |
|---|---|---|---|---|---|---|
| CPU (FFTW) | 51 | 51 | 275 | 197 | 1088 | 2936 |
| AX8850 (M5Stack LLM-8850) | **15** (3.4x) | **15** (3.4x) | **72** (3.8x) | **66** (3.0x) | **206** (5.3x) | **581** (5.1x) |
| Hailo-8 (AI HAT+) | **23** (2.2x) | **23** (2.2x) | **88** (3.1x) | **72** (2.7x) | **411** (2.6x) | - (*1) |
| Hailo-10H (AI HAT+ 2) | **63** (0.8x) | **62** (0.8x) | **133** (2.1x) | **116** (1.7x) | **504** (2.2x) | **1134** (2.6x) |

(\*1) No model (too large for the Hailo-8).

### ROCK 5C (RK3588)
- Cortex-A76 + Cortex-A55, search time per signal (ms) and speed-up to the CPU on
the same cores (x):

| Method | L1CA (8 MHz) | B1I (8 MHz) | E1B (8 MHz) | L5I (24 MHz) | L1CP (8 MHz) | L2CM (8 MHz) |
|---|---|---|---|---|---|---|
| CPU (FFTW, A76) | 52 | 52 | 306 | 203 | 1123 | 2918 |
| RK3588 NPU (A76) | **45** (1.1x) | **45** (1.2x) | **165** (1.9x) | **132** (1.5x) | **513** (2.2x) | **1231** (2.4x) |
| CPU (FFTW, A55) | 263 | 259 | 1536 | 1002 | 5557 | 13213 |
| RK3588 NPU (A55) | **65** (4.0x) | **65** (4.0x) | **207** (7.4x) | **181** (5.5x) | **645** (8.6x) | **1584** (8.3x) |

- The NPU detects the same signals as the CPU, with the sensitivity within 0.06 dB.
- In the receiver, the real-time limit of the tracked signals on the Raspberry Pi 5 with AX8850 increases from about 175 to about 230.

## **Supported NPUs**

| NPU | Plugin | Runtime | Model generator |
|---|---|---|---|
| Axera AX8850 (M5Stack LLM-8850, M.2 PCIe) | `libsdr_acc_axcl.so` | AXCL 3.6 | `lib/build/gen_axcl_acq.py` (Pulsar2 8.0) |
| Hailo-8 (Raspberry Pi AI HAT+) | `libsdr_acc_hailo.so` | HailoRT 4.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Hailo-10H (Raspberry Pi AI HAT+ 2) | `libsdr_acc_hailo.so` | HailoRT 5.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Rockchip RK3588 NPU | `libsdr_acc_rknn.so` | RKNN runtime 2.3 | `lib/build/gen_rknn_acq.py` (RKNN-Toolkit2) |

The models depend on the NPU, the runtime version and the sampling rate. The
models for the 24 MHz configuration above are in the
[releases](https://github.com/taroz/PocketSDR-NPU/releases). Unzip
`pocketsdr-npu-models-<npu>.zip` in `lib/linux`.

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

IF data of the results (Pocket SDR FE 4CH, 24 MHz, 300 s):

- [L1L2L5E5b_24MHz_20261009_0635_300s.bin](https://www.taroz.net/ifdata/L1L2L5E5b_24MHz_20261009_0635_300s.bin) (14.4 GB)
- [L1L2L5E5b_24MHz_20261009_0635_300s.bin.tag](https://www.taroz.net/ifdata/L1L2L5E5b_24MHz_20261009_0635_300s.bin.tag)
- [L1L2L5E5b_24MHz_20261009_0635_300s.md5](https://www.taroz.net/ifdata/L1L2L5E5b_24MHz_20261009_0635_300s.md5) (MD5 checksums)

## **Changes from Pocket SDR**

- Acquisition plugin interface (`src/sdr_acc.c`, `src/sdr_acc.h`, API version
  2 with the optional peak search `search_code_peak`) and the plugins for
  AX8850, RK3588 and Hailo (`src/sdr_acc_axcl.cpp`, `src/sdr_acc_rknn.cpp`,
  `src/sdr_acc_hailo.cpp`)
- Receiver options for SBCs (`-dec`, `-trk_guard`, `-rcv_prio`, `-cpu_acq`,
  `-cpu_trk`)

Branches: `main` (this version), `dev` (development history with the other
evaluated methods).

## **Paper**

T. Suzuki, NPU-Accelerated GNSS Signal Acquisition, arXiv:2611.18186, 2026.
[arXiv](https://arxiv.org/abs/2611.18186)

## **Acknowledgments**

The author is deeply grateful to Mr. Tomoji Takasu for developing
[Pocket SDR](https://github.com/tomojitakasu/PocketSDR), on which this work is
based, and for making it publicly available as open source.

## **License**

BSD 2-clause license, the same as Pocket SDR (see [LICENSE.txt](LICENSE.txt)).
Pocket SDR is developed by T. Takasu.
