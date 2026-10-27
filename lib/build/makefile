#
#  makefile for Pocket SDR external shared and static libraries
#
#! To use FFTW3 instead of PocketFFT, install FFTW3 and build with USE_FFTW=1.
#! To build NPU backend plugin (OpenVINO), build with USE_NPU=1.
#! To build Rockchip NPU backend plugin (RKNN), build with USE_RKNN=1.
#! To build Axera NPU backend plugin (AXCL), build with USE_AXCL=1.

USE_FFTW ?= 0
USE_NPU ?= 0
USE_RKNN ?= 0
USE_AXCL ?= 0

all:
	make -f librtk.mk
ifneq ($(USE_FFTW),1)
	make -f libpocketfft.mk
endif
	make -f libsdr.mk
ifeq ($(USE_NPU),1)
	make -f libsdr_acc_ov.mk
endif
ifeq ($(USE_RKNN),1)
	make -f libsdr_acc_rknn.mk
endif
ifeq ($(USE_AXCL),1)
	make -f libsdr_acc_axcl.mk
endif
clean:
	make -f librtk.mk clean
	make -f libpocketfft.mk clean
	make -f libsdr.mk clean
	make -f libsdr_acc_ov.mk clean
	make -f libsdr_acc_rknn.mk clean
	make -f libsdr_acc_axcl.mk clean
install:
	make -f librtk.mk install
ifneq ($(USE_FFTW),1)
	make -f libpocketfft.mk install
endif
	make -f libsdr.mk install
ifeq ($(USE_NPU),1)
	make -f libsdr_acc_ov.mk install
endif
ifeq ($(USE_RKNN),1)
	make -f libsdr_acc_rknn.mk install
endif
ifeq ($(USE_AXCL),1)
	make -f libsdr_acc_axcl.mk install
endif
