# UEGAN-MPS: MPS-Optimized UEGAN Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains an enhanced implementation of **UEGAN (Unsupervised Enhancement GAN)** optimized for Apple's **Metal Performance Shaders (MPS)** environment, built on top of the original UEGAN codebase by [eezkni](https://github.com/eezkni/UEGAN). The modifications, contributed by *duhyeon kim*, focus on MPS compatibility, performance optimization, and modern PyTorch practices.

## 📖 About
This project extends the original UEGAN, a GAN-based model for unsupervised image enhancement, to support MPS devices (e.g., Apple Silicon GPUs) alongside CUDA and CPU. Key improvements include device-agnostic code, memory optimization, and compatibility with PyTorch's latest features. All changes are documented with *duhyeon kim* comments in the codebase.

### Original Source
This implementation is based on:
- **Paper**: [Unsupervised Image Enhancement Using GANs](https://arxiv.org/abs/2012.15020) *(Replace with the correct arXiv link if available)*  
- **Code**: [eezkni/UEGAN](https://github.com/eezkni/UEGAN)  
We acknowledge the original authors for their foundational work.

> Major changes for MPS support are clearly indicated with `# duhyeon kim` comments in the code. Please check these comments to understand all MPS-specific modifications.

## 🚀 Key Features
- **MPS Support**: Automatic device detection (MPS/CUDA/CPU) for seamless execution on Apple Silicon.  
- **Optimized Perceptual Loss**: VGG19-based loss adapted for MPS with dynamic `InstanceNorm2d` and no `AveragePool2d`.  
- **Modern PyTorch**: Replaced `Swish` with `SiLU`, used `nn.Identity`, and avoided `.data` in favor of `.detach()`.  
- **Memory Efficiency**: Added `torch.mps.empty_cache()` and dictionary reuse with `.clear()`.  
- **Improved Data Handling**: Enhanced `InputFetcher` with device-aware batch processing and robust dataset path handling.  
- **Code Readability**: Leveraged `pathlib` for path management and improved logging/print statements.

*All modifications are marked with `# duhyeon kim` comments in the code.*


## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dudududukim/UEGAN-mps.git
   cd UEGAN-mps
   ```
2. Install dependencies:
3. Prepare your dataset (e.g., low-quality and high-quality image pairs).
4. Update dataset paths in the configuration file or script.

## ▶️ Usage(device is automatically selected) -> if you are using CUDA GPU please use the original code for stability
Train the model:
```bash
python main.py --mode train --version UEGAN-ver_name --use_tensorboard True --is_test_nima True --is_test_psnr_ssim True
```

Generate enhanced images:
```bash
python main.py --mode test --version UEGAN-ver_name --pretrained_model 100 --is_test_nima True --is_test_psnr_ssim False
```

## 📝 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

The original UEGAN codebase by [eezkni](https://github.com/eezkni/UEGAN) is subject to its own license. Please review it before using this repository.

## 🙏 Acknowledgments
- **Original Authors**: [eezkni](https://github.com/eezkni/UEGAN) for the UEGAN implementation and paper.
- **Contributor**: *duhyeon kim* for MPS support, optimizations, and code enhancements.

## 📬 Contact
For questions or suggestions, feel free to open an issue or contact <kdhluck@naver.com>.

---

*Happy enhancing! ✨*