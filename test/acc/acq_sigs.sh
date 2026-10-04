#!/bin/bash
#  signal set for acquisition regression/benchmark on FE 8CH 16 MHz IF data
#  (L1B1G1L2G2L5E5bL6_*_16MHz.bin, RAW32)
FO=1575.42,1561,1602,1227.6,1246,1176.45,1207.14,1273
SIGS="-sig L1CA -prn 1-32 -rfch 1 -sig E1B -prn 1-36 -rfch 1 \
-sig B1I -prn 1-63 -rfch 2 -sig G1CA -prn -7-6 -rfch 3 -sig G2CA -prn -7-6 -rfch 5 \
-sig L5I -prn 1-32 -rfch 6 -sig E5AI -prn 1-36 -rfch 6 -sig E5BI -prn 1-36 -rfch 7 \
-sig L6D -prn 193-202 -rfch 8 -sig E6B -prn 1-36 -rfch 8"
