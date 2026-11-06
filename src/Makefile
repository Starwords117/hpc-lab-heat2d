# 学员构建: NVIDIA 用 make (需要 nvcc); AMD/ROCm 机器用 make hip (hipcc)
#
# -arch 对照 (CUDA >= 11.5 可用默认的 -arch=native 自动探测):
#   RTX 2060 -> sm_75   RTX 3050/3050Ti/3060 -> sm_86   RTX 4050/4060 -> sm_89

NVCC  ?= nvcc
ARCH  ?= native
NVFLAGS = -O3 -arch=$(ARCH)
CXX   ?= g++
HIPCFLAGS = -O3 -D__HIP_PLATFORM_AMD__ -lamdhip64

all: warmup serial naive

warmup: warmup_matadd.cu gpu_compat.h
	$(NVCC) $(NVFLAGS) -o $@ $<

serial: heat_serial.cpp
	$(CXX) -O3 -march=native -o $@ $<

naive: heat_naive.cu gpu_compat.h
	$(NVCC) $(NVFLAGS) -o $@ $<

# AMD/ROCm 备用路径 (同一份源码, gpu_compat.h 提供 API 别名)
hip: warmup_hip serial naive_hip

warmup_hip: warmup_matadd.cu gpu_compat.h
	hipcc $(HIPCFLAGS) -o warmup_hip $<

naive_hip: heat_naive.cu gpu_compat.h
	hipcc $(HIPCFLAGS) -o naive_hip $<

.PHONY: all hip
