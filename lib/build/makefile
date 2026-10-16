#
#  makefile for Pocket SDR external shared and static libraries
#
#! To use FFTW3 instead of PocketFFT, install FFTW3 and build with USE_FFTW=1.
#! To build the acquisition plugins, build with USE_HAILO=1 (Hailo NPU,
#! HailoRT) or USE_RKNN=1 (Rockchip NPU, RKNN runtime).

USE_FFTW ?= 0
USE_HAILO ?= 0
USE_RKNN ?= 0

all:
	make -f librtk.mk
ifneq ($(USE_FFTW),1)
	make -f libpocketfft.mk
endif
	make -f libsdr.mk
ifeq ($(USE_HAILO),1)
	make -f libsdr_acc_hailo.mk
endif
ifeq ($(USE_RKNN),1)
	make -f libsdr_acc_rknn.mk
endif
clean:
	make -f librtk.mk clean
	make -f libpocketfft.mk clean
	make -f libsdr.mk clean
	make -f libsdr_acc_hailo.mk clean
	make -f libsdr_acc_rknn.mk clean
install:
	make -f librtk.mk install
ifneq ($(USE_FFTW),1)
	make -f libpocketfft.mk install
endif
	make -f libsdr.mk install
ifeq ($(USE_HAILO),1)
	make -f libsdr_acc_hailo.mk install
endif
ifeq ($(USE_RKNN),1)
	make -f libsdr_acc_rknn.mk install
endif
