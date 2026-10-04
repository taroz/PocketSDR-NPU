#
#  makefile for Pocket SDR accelerator backend plugin for OpenVINO (Intel NPU)
#  (libsdr_acc_ov.so loaded by sdr_acc_init("npu") at runtime)
#
#! Install OpenVINO archive and set OV_DIR (or source setupvars.sh).

SRC = ../../src
OV_DIR ?= $(if $(INTEL_OPENVINO_DIR),$(INTEL_OPENVINO_DIR),$(HOME)/opt/openvino)
OV_LIB = $(OV_DIR)/runtime/lib/intel64

CXX = g++
INSTALL = ../linux
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall -g -I$(SRC) -I$(OV_DIR)/runtime/include
LDLIBS = -L$(OV_LIB) -lopenvino -Wl,-rpath,$(OV_LIB)
ifeq ($(shell uname -m),x86_64)
    CXXFLAGS += -mavx2 -mfma -mf16c
endif

TARGET = libsdr_acc_ov.so

all : $(TARGET)

libsdr_acc_ov.so: sdr_acc_ov.o
	$(CXX) -shared -o $@ $< $(LDLIBS)
sdr_acc_ov.o : $(SRC)/sdr_acc_ov.cpp $(SRC)/sdr_acc.h
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(TARGET) sdr_acc_ov.o

install:
	cp $(TARGET) $(INSTALL)
