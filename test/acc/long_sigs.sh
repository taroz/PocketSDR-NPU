#!/bin/bash
#  signal set of direct acquisition of long codes (L1C 10 ms, L2CM 20 ms)
#  on FE 8CH 16 MHz IF data (L1B1G1L2G2L5E5bL6_*_16MHz.bin, RAW32)
FO=1575.42,1561,1602,1227.6,1246,1176.45,1207.14,1273
SIGS="-sig L1CP -prn 1,2,3,4,7,8,10,16,21,27,30 -rfch 1 \
-sig L2CM -prn 1,2,3,4,7,8,10,16,21,27,30 -rfch 4"
WARM="-sig L1CP -prn 1 -rfch 1 -sig L2CM -prn 1 -rfch 4"
