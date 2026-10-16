#
#  makefile for Pocket SDR accelerator backend plugin for Rockchip NPU
#  (libsdr_acc_rknn.so loaded by sdr_acc_init("rknn") at runtime)
#
#! Install RKNN runtime (librknnrt.so, rknn_api.h, rknn_matmul_api.h, e.g.
#! package rknpu2-rk3588) and set RKNN_DIR if not in /usr.

SRC = ../../src
RKNN_DIR ?= /usr

CXX = g++
INSTALL = ../linux
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall -g -mcpu=native -I$(SRC) \
    -I$(RKNN_DIR)/include
LDLIBS = -L$(RKNN_DIR)/lib -lrknnrt -lpthread

TARGET = libsdr_acc_rknn.so

all : $(TARGET)

libsdr_acc_rknn.so: sdr_acc_rknn.o
	$(CXX) -shared -o $@ $< $(LDLIBS)
sdr_acc_rknn.o : $(SRC)/sdr_acc_rknn.cpp $(SRC)/sdr_acc.h
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(TARGET) sdr_acc_rknn.o

install:
	cp $(TARGET) $(INSTALL)
