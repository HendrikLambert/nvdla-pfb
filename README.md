# NVDLA Polyphase Filter Bank (PFB)

This repository accompanies the thesis *An Optimized Polyphase Filter Bank for NVDLA*. A polyphase filter bank is a filtering step often found in radio astronomy. The aim of this project is to research if (followed by how performant) this filtering step can be implemented on NVDLA, which is a deep learning accelerator for edge computing made my NVIDIA.

The project consists of code to generate a polyphase filter bank as a PyTorch model (`pytorch_model` subfolder). This model can then be converted to the ONNX file format. This in turn, can be converted to a NVDLA loadable, which can be benchmarked or simply executed with the code found in the `runtime` folder.

## Requirements
- NVIDIA JetPack 6.1 (CUDA 12.6, g++ 10.4)
- TensorRT 10.7 - installed separately, as TensorRT 10.3 contained in JetPack 6.1 contains a bug.
- [PowerSensor3 v1.6.1](https://github.com/nlesc-recruit/PowerSensor3) 
- Python 3.10, with PyTorch 2.6.0, ONNX 1.17 and NumPy 1.26.

## Project Structure

### `pytorch_model/`
Contains PyTorch implementations of PFB and tools for model export.

- `main.py` - Main commandline file for building, testing, and exporting models
- `model_helper.py` - Helper functions for batch model generation
- `onnx_helper.py` - ONNX export utilities
- `modules/` - PyTorch module implementations (PFB FFT, PFB DFT, FIR, etc.)
- `tests/` - Unit tests for the PyTorch modules

### `runtime/`
C++ runtime for executing NVDLA loadables with performance benchmarking and power measurement.

- `benchmark.cpp/hpp` - Main benchmarking framework with power sensor integration
- `loadable.cpp/hpp` - NVDLA loadable management and execution
- `cudla_runtime.cpp/hpp` - CUDA/CUDLA runtime wrapper
- `simple.cpp` - Simple runtime for testing individual loadables
- `main.cpp` - Entry point for the benchmark runtime executable
-  `Makefile` - Build system for compiling runtime executables

### `analysis/`
Jupyter notebooks for analysing benchmark results.

- `analysis.ipynb` - Performance analysis and visualization
- `accuracy.ipynb` - Model accuracy evaluation

### `reference/`
Reference implementations for validation.

- `polyphase-filter-bank-generator/` - Slightly modified reference implementation by Rob van Nieuwpoort (https://github.com/NLeSC/polyphase-filter-bank-generator)

### `lb_reveng/`
Loadable reverse engineering tools for inspecting NVDLA loadable files. Leftover code from earlier attempts at understanding the NVDLA file format.

- `lb_helper.py` - Command-line interface for loadable inspection
- `lb_printer.py` - Loadable file parser and printer
- `loadable.fbs` - FlatBuffers schema for NVDLA loadables

### `legacy/`
Older model implementations using TensorFlow/TensorRT.

- `tensorflow_model.py` - Legacy TensorFlow implementation
- `tensorrt_model.py` - Legacy TensorRT implementation
