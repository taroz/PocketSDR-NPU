# **PocketSDR-NPU - GNSS Signal Acquisition on NPUs of Single-Board Computers**

**PocketSDR-NPU** runs the GNSS signal acquisition (parallel code search) of
[**Pocket SDR**](https://github.com/tomojitakasu/PocketSDR) on the NPU of a
low-power single-board computer (SBC), e.g. Raspberry Pi 5 with Hailo-8 or
an RK3588 board. The NPU makes the acquisition faster than the CPU (FFTW),
2-3 times on the Raspberry Pi 5 with Hailo-8 and up to 9 times on the
Cortex-A55 of RK3588, and reduces its CPU time, so the receiver searches more
signals with the same CPU and locks them faster. The long codes (L1CP 10 ms, L2CM 20 ms)
can be acquired directly over the full Doppler range.

The receiver itself (tracking, navigation data, PVT) is Pocket SDR. For Pocket
SDR, its RF front-ends and its documents, see the
[**original repository**](https://github.com/tomojitakasu/PocketSDR).

---

## **How It Works**

- The parallel code search is a DFT-based circular correlation of the IF data
  and the code for each Doppler bin. With the Doppler bins spaced by fs / M
  (M: DFT size), all bins share one DFT of the IF data shifted by the bin.
- The DFT is computed on the NPU as a four-step DFT (matrix products, twiddle
  factors and transposes) in INT8 (the last IDFT and the power in 16-bit or
  FP16). One inference processes a chunk of Doppler bins.
- The acquisition is a plugin (`libsdr_acc_<name>.so`) loaded at runtime by
  `-acc <name>`. Signals without a model are searched by the CPU.

## **Results**

Acquisition time per signal (ms, all PRNs, mean of 2 runs), 24 MHz IF data of
Pocket SDR FE 4CH (`pocket_L1L2L5E5b_24MHz.conf`), the narrowband signals
decimated to 8 MHz. CPU: the same search on the CPU with FFTW.

| Signal (fs) | Pi 5 CPU | Pi 5 + Hailo-8 | RK3588 A76 CPU | RK3588 A76 + NPU | RK3588 A55 CPU | RK3588 A55 + NPU |
|---|---|---|---|---|---|---|
| L1CA (8 MHz) | 51 | 23 | 53 | 46 | 263 | 66 |
| E1B (8 MHz) | 280 | 88 | 295 | 162 | 1531 | 206 |
| L5I (24 MHz) | 198 | 73 | 204 | 123 | 972 | 174 |
| L1CP (8 MHz, direct) | 1137 | 414 | 1036 | 485 | 5483 | 619 |
| L2CM (8 MHz, direct) | 3058 | - (CPU) | 2745 | 1170 | 12732 | 1522 |

- The detections are the same as the CPU. The sensitivity (C/N0 at the
  probability of detection 0.5) differs by 0.01-0.14 dB.
- In the receiver (300 s, fixed sets of 140-260 signals), the NPU searches
  1.4-1.9 times more signals with the same CPU time and locks 1.2-1.3 times
  more signals by 30 s.

## **Supported NPUs**

| NPU | Plugin | Runtime | Model generator |
|---|---|---|---|
| Hailo-8 (Raspberry Pi AI HAT+) | `libsdr_acc_hailo.so` | HailoRT 4.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Hailo-10H (Raspberry Pi AI HAT+ 2) | `libsdr_acc_hailo.so` | HailoRT 5.x | `lib/build/gen_hailo_acq.py` (Hailo Dataflow Compiler) |
| Rockchip RK3588 NPU | `libsdr_acc_rknn.so` | RKNN runtime 2.3 | `lib/build/gen_rknn_acq.py` (RKNN-Toolkit2) |

The models depend on the NPU, the runtime version and the sampling rate. The
models for the 24 MHz configuration above are in the release assets.

| Model directory | Models (`acq_M<M>_K<Kc>_R<R>`) |
|---|---|
| `lib/linux/hailo8` | Hailo-8 HEFs (no L2CM, too large for the Hailo-8) |
| `lib/linux/hailo` | Hailo-10H HEFs |
| `lib/linux/rknn` | RK3588 INT8 RKNN models |

## **Build**

Requirements: Linux (ARM64), FFTW3 (`libfftw3-dev`), and HailoRT or the RKNN
runtime.

```
cd lib/build
make USE_SOAPY=0 USE_FFTW=1 USE_HAILO=1        # Hailo (or USE_RKNN=1 for RK3588)
make USE_SOAPY=0 USE_FFTW=1 USE_HAILO=1 install
cd ../../app/pocket_trk
make USE_SOAPY=0 USE_FFTW=1 && make USE_SOAPY=0 USE_FFTW=1 install
```

## **Usage**

```
# Raspberry Pi 5 + Hailo-8
pocket_trk -acc hailo:hef=../lib/linux/hailo8,batch=16 -dec 1:3:7.42,1:3:-6.902,2:3 \
    -trk_guard 20 -rcv_prio 10 -c pocket_L1L2L5E5b_24MHz.conf [-sig ... -prn ...] [file]

# RK3588
pocket_trk -acc rknn -dec 1:3:7.42,1:3:-6.902,2:3 -trk_guard 20 -rcv_prio 10 ...
```

| Option | Description |
|---|---|
| `-acc backend` | Acquisition backend: `cpu` (default), `hailo[:hef=dir,batch=n]`, `rknn[:cores=n,model=dir]` |
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

- Acquisition plugin interface (`src/sdr_acc.c`, `src/sdr_acc.h`) and the
  plugins for Hailo and RK3588 (`src/sdr_acc_hailo.cpp`, `src/sdr_acc_rknn.cpp`)
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
