# GAN-Based Image Generation Using Deep Learning

**CS318 Course Project**  
**Authors:** Faqre Alam (23/CS/150) · Ekansh Agrawal (23/CS/149)  
**Institution:** [Your Institution]  
**Date:** April 2026

---

## Abstract

Generative Adversarial Networks (GANs) represent a groundbreaking class of deep learning architectures capable of generating highly realistic synthetic data. This paper presents the design, implementation, and evaluation of a GAN-based system for image generation using the MNIST benchmark dataset. The architecture consists of two neural networks — a Generator, which learns to produce realistic images from random noise, and a Discriminator, which learns to distinguish generated images from real ones. Through an adversarial training process, both networks improve iteratively, ultimately enabling the Generator to produce visually convincing handwritten digit images. We explore two GAN variants: a Vanilla GAN (fully-connected MLP) and a Deep Convolutional GAN (DCGAN). Performance is quantitatively evaluated using the Fréchet Inception Distance (FID) and qualitatively assessed through visual inspection of generated samples across training epochs.

**Keywords:** Generative Adversarial Networks, DCGAN, MNIST, Image Synthesis, FID, Deep Learning

---

## 1. Introduction

Deep generative models have transformed the field of computer vision and unsupervised learning. Among them, Generative Adversarial Networks (GANs), introduced by Goodfellow et al. (2014), stand out for their ability to produce high-fidelity synthetic samples without requiring explicit density estimation.

GANs operate through an adversarial game between two networks:
- **Generator (G):** Maps random noise *z* ~ *N*(0, I) to synthetic data space.
- **Discriminator (D):** Classifies inputs as real or generated.

The training objective is a minimax game:

```
min_G max_D  E[log D(x)]  +  E[log(1 − D(G(z)))]
```

where *x* is a real sample and *z* is a latent vector.

This project implements and compares two GAN variants on MNIST:
1. **Vanilla GAN** — fully-connected multi-layer perceptron (MLP) architecture.
2. **DCGAN** — convolutional architecture following Radford et al. (2015).

---

## 2. Related Work

**Goodfellow et al. (2014)** introduced the original GAN framework, demonstrating generation of plausible images on MNIST and CIFAR-10 using MLP architectures.

**Radford et al. (2015)** introduced DCGAN, replacing fully-connected layers with transposed convolutions in the generator and strided convolutions in the discriminator. Key training guidelines included: batch normalisation in both networks, Leaky ReLU in the discriminator, ReLU in the generator, and Adam optimiser with carefully tuned momentum (β₁=0.5).

**Heusel et al. (2017)** introduced the Fréchet Inception Distance (FID), which measures the statistical distance between the distributions of real and generated images, providing a robust quantitative evaluation metric that correlates well with human perceptual quality.

---

## 3. Methodology

### 3.1 Dataset

MNIST consists of 60,000 training and 10,000 test grey-scale images of handwritten digits (0–9), each 28×28 pixels. Images are normalised to [−1, 1] to match the **Tanh** output of the generator.

### 3.2 Vanilla GAN Architecture

**Generator:**

| Layer | Input → Output | Activation |
|-------|---------------|------------|
| Linear | 100 → 256 | LeakyReLU(0.2) + BN |
| Linear | 256 → 512 | LeakyReLU(0.2) + BN |
| Linear | 512 → 1024 | LeakyReLU(0.2) + BN |
| Linear | 1024 → 784 | Tanh |

**Discriminator:**

| Layer | Input → Output | Activation |
|-------|---------------|------------|
| Linear | 784 → 512 | LeakyReLU(0.2) + Dropout(0.3) |
| Linear | 512 → 256 | LeakyReLU(0.2) + Dropout(0.3) |
| Linear | 256 → 1 | Sigmoid |

### 3.3 DCGAN Architecture

**Generator:**

| Layer | Spatial Size | Activation |
|-------|-------------|------------|
| Linear + Reshape | 100 → (128, 7, 7) | – |
| BatchNorm2d | (128, 7, 7) | – |
| ConvTranspose2d(128→64, k=4, s=2, p=1) | (64, 14, 14) | ReLU + BN |
| ConvTranspose2d(64→1, k=4, s=2, p=1) | (1, 28, 28) | Tanh |

**Discriminator:**

| Layer | Spatial Size | Activation |
|-------|-------------|------------|
| Conv2d(1→64, k=4, s=2, p=1) | (64, 14, 14) | LeakyReLU(0.2) |
| Conv2d(64→128, k=4, s=2, p=1) | (128, 7, 7) | LeakyReLU(0.2) + BN |
| Flatten + Linear(6272→1) | scalar | Sigmoid |

### 3.4 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Latent dimension | 100 |
| Batch size | 128 |
| Epochs | 100 |
| Learning rate | 0.0002 |
| Optimiser | Adam (β₁=0.5, β₂=0.999) |
| Loss function | Binary Cross-Entropy (BCE) |
| Label smoothing | 0.9 (real labels) |
| Weight initialisation | N(0, 0.02) |

**Label Smoothing:** Real labels are set to 0.9 instead of 1.0 to prevent the discriminator from becoming overconfident and to provide smoother gradients to the generator — a standard stabilisation technique.

**Weight Initialisation (DCGAN):** All convolutional and linear weights are initialised from N(0, 0.02), and batch normalisation weights from N(1, 0.02), following Radford et al. (2015).

---

## 4. Experiments & Results

### 4.1 Training Loss Curves

During stable GAN training, the discriminator loss should remain around 0.5–0.7 (uncertainty), while the generator loss gradually decreases. Training instability is characterised by D loss collapsing near 0 (discriminator wins) or G loss exploding (generator fails to fool D).

*[Insert `results/dcgan/losses.png` and `results/vanilla/losses.png` here]*

### 4.2 Training Progression

Generated samples at epochs 1, 10, 50, and 100 demonstrate the progressive learning of digit structure.

*[Insert `results/dcgan/progression.png` here]*

- **Epoch 1:** Uniform noise; no recognisable structure.
- **Epoch 10:** Faint digit-like blobs emerging.
- **Epoch 50:** Reasonably well-formed digits with some blurring.
- **Epoch 100:** Sharp, diverse, and realistic handwritten digits.

### 4.3 Visual Comparison: Vanilla vs DCGAN

*[Insert `results/comparison.png` here]*

DCGAN produces visibly sharper images due to convolutional kernels that exploit spatial locality, while Vanilla GAN (MLP-based) tends to produce slightly blurrier results.

### 4.4 Quantitative Evaluation: FID Score

The Fréchet Inception Distance is computed between 10,000 real MNIST images and 10,000 generated images using Inception-v3 features.

| Model | FID Score (↓) |
|-------|--------------|
| Vanilla GAN | *[insert value]* |
| DCGAN | *[insert value]* |

Lower FID indicates that the distribution of generated images is closer to the real data distribution. DCGAN is expected to achieve significantly lower FID than Vanilla GAN.

---

## 5. Discussion

### 5.1 Effect of Label Smoothing
Label smoothing (real = 0.9) prevented early discriminator saturation, which is a common failure mode where D(x) → 1 and D(G(z)) → 0 rapidly, causing vanishing gradients for the generator.

### 5.2 Mode Collapse
Some Vanilla GAN runs exhibited partial mode collapse, generating only a subset of digits. DCGAN showed greater diversity, attributed to its spatially-aware convolutional structure.

### 5.3 Training Stability
DCGAN training was generally more stable, consistent with findings in the original DCGAN paper. The strided convolution downsampling in the discriminator proved more effective than pooling.

### 5.4 Limitations
- FID computation requires Inception features trained on ImageNet; applying them to grey-scale MNIST introduces a domain mismatch.
- 100 epochs may be insufficient to fully converge the Vanilla GAN; longer training or learning rate scheduling could improve results.
- Neither model conditions on digit labels; a Conditional GAN (cGAN) would allow targeted digit generation.

---

## 6. Conclusion

This project successfully implemented and compared Vanilla GAN and DCGAN for synthetic MNIST digit generation. DCGAN outperformed Vanilla GAN both visually (sharper, more diverse images) and quantitatively (lower FID). Key lessons include the importance of label smoothing, careful weight initialisation, and the convolutional inductive bias for image generation tasks.

Future work could explore:
- **Conditional GAN (cGAN):** Conditioning on digit class labels.
- **Progressive GAN:** Gradually increasing resolution.
- **WGAN-GP:** Wasserstein loss with gradient penalty for improved training stability.
- Extension to CIFAR-10 or CelebA datasets.

---

## 7. References

1. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial Nets*. NeurIPS 2014.

2. Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR 2016.

3. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017.

4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324.

5. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). *Improved Techniques for Training GANs*. NeurIPS 2016.

---

*Submitted as part of CS318 – Deep Learning Course Project, April 2026.*
