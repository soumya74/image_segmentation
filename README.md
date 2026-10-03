# Edge-Optimized Semantic Segmentation on Intel Core Ultra NPU

## Overview
This repository contains a full-stack deep learning pipeline for semantic segmentation, demonstrating the lifecycle from PyTorch training to silicon-specific hardware acceleration. The project trains a DeepLabV3-ResNet50 model on the Cityscapes dataset and rigorously optimizes its computational graph for edge deployment on an Intel Core Ultra NPU using OpenVINO.

## Architecture & Training Pipeline
To accommodate standard Colab GPU memory limits without encountering OOM crashes, the pipeline implements aggressive data transformations and dynamic evaluation:
* **Downscaled Data**: Original high-resolution Cityscapes images were downscaled to 256x512 tensors to stabilize memory consumption during batch processing.
* **Pre-trained Backbone**: Initialized a ResNet50 backbone with pre-trained COCO weights, applying differential learning rates (10x slower on the backbone) to preserve low-level edge detection filters.
* **Custom Metrics**: Evaluated performance using Mean Intersection over Union (mIoU) natively, explicitly ignoring the "void" class to prevent artificially skewed accuracy metrics.
* **Training Convergence**: Over 5 epochs, the training loss steadily dropped from 0.1751 to 0.0064, and validation loss decreased from 0.0399 to 0.0175. The custom dynamic checkpointing script successfully monitored the validation metrics, preventing regression and saving the peak weights at a 51.50% validation mIoU.

## ONNX Export & OpenVINO IR Compilation
Bridging the gap between a dynamic Python training environment and a static C++ edge device required stripping framework overhead and locking the network topology:
* **ONNX Export**: Exported the PyTorch computational graph to ONNX (Opset 18) utilizing aggressive constant folding to hardcode frozen Batch Normalization statistics directly into the graph. A custom wrapper was applied to strip dictionary mappings and force a raw tensor output.
* **FP16 Compression**: Compiled the universal ONNX graph into Intel's OpenVINO Intermediate Representation (IR), deliberately compressing the 32-bit floating-point weights to FP16. This doubled compute density and slashed memory bus traffic.

## Performance Analysis & Hardware Profiling
The final deployment script bypasses general-purpose processing entirely, routing the segmentation matrix mathematics directly to the Intel Core Ultra NPU's dedicated spatial compute engines. 

Benchmarking the exact same DeepLabV3 graph across different execution backends demonstrates a **17.86x total execution speedup**:

| Hardware / Execution Backend | Average Latency | Throughput |
| :--- | :--- | :--- |
| **PyTorch (Native CPU Baseline)** | 1597.15 ms | 0.63 FPS |
| **OpenVINO (CPU Optimized)**| 886.74 ms | 1.13 FPS |
| **OpenVINO (Intel Core Ultra NPU)** | **89.40 ms** | **11.19 FPS** |

### Conclusion
Relying on unoptimized PyTorch for edge execution was computationally unviable, taking over 1.5 seconds per frame. By converting the model to OpenVINO IR and routing the compute to the dedicated Intel NPU, latency plummeted to under 90 milliseconds. This unlocked near real-time 11.19 FPS performance for a complex segmentation network while leaving the primary system CPU and GPU cores completely unburdened.