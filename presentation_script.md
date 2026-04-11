# 🎙️ CS318 GAN Project — Complete Presentation Guide

This document contains a slide-by-slide breakdown for your presentation. For each slide, you will find what to put on the screen (**Visuals/Content**), exactly what you should say (**Speech**), and questions the examiner might ask you (**Expected Questions & Answers**).

---

## Slide 1: Title Slide
**Visuals/Content:**
*   **Title:** GAN-Based Image Generation (CS318)
*   **Sub-title:** A Comparative Study from Vanilla GAN to Conditional DCGAN on MNIST and Fashion-MNIST
*   **Authors:** Faqre Alam (23/CS/150) & Ekansh Agrawal (23/CS/149)
*   **Logos:** College/University Logo (if applicable)

**Speech:**
> "Good morning/afternoon everyone. Today, my partner Ekansh and I are excited to present our deep learning project on Generative Adversarial Networks, or GANs. Our project focuses on generating images by comparing three different architectural approaches: the fully-connected Vanilla GAN, the Convolutional DCGAN, and an advanced Conditional DCGAN. We conducted our experiments across both the classical MNIST dataset and the more complex Fashion-MNIST dataset."

**Expected Questions & Answers:**
*   **Q:** *Why did you extend the project to Fashion-MNIST?*
    *   **A:** MNIST is an excellent proving ground, but it's fundamentally very easy for modern convolutional networks to solve. We integrated Fashion-MNIST—which shares the same 28x28 grayscale dimensional footprint—to prove that our generative pipeline is robust enough to learn much more complex spatial topologies like clothing items without modifying the network dimensions.

---

## Slide 2: What is a GAN? (The Core Idea)
**Visuals/Content:**
*   **Diagram:** The Generator (Counterfeiter) vs. Discriminator (Detective) flowchart. 
*   **Text Bullet:** A zero-sum game between two neural networks.
*   **Text Bullet:** Generator mapping random noise $z \sim \mathcal{N}(0,1)$ to images.

**Speech:**
> "Before diving into the architectures, let's briefly recap what a GAN is. Think of it as a game of cat and mouse. We have two neural networks: a Generator, which acts like a counterfeiter trying to print fake money, and a Discriminator, which acts as a detective trying to spot the fakes. The Generator takes random noise as input and shapes it into an image. The Discriminator looks at both real images from our dataset and fake images from the Generator, and outputs a probability of whether the image is real. Through this pure competition, without ever seeing a real image directly, the Generator eventually learns to create incredibly realistic outputs."

**Expected Questions & Answers:**
*   **Q:** *Does the Generator ever see the real data?*
    *   **A:** No, never. It only receives gradient feedback from the Discriminator. It learns entirely by analyzing how the Discriminator reacted to its generated fakes.
*   **Q:** *What happens if the Discriminator becomes too good too quickly?*
    *   **A:** If the Discriminator is perfect, it easily rejects all fakes with high confidence. The loss goes to zero, the gradients vanish, and the Generator gets no useful feedback to improve. This is why keeping them balanced is important.

---

## Slide 3: Vanilla GAN Architecture
**Visuals/Content:**
*   **Diagram:** Flowchart of Vanilla Generator and Discriminator (MLPs).
*   **Text:** 100D noise $\rightarrow$ Fully Connected Layers $\rightarrow$ 784D vector $\rightarrow$ 28x28 Image.
*   **Text:** Treats images as flat vectors. No spatial awareness.

**Speech:**
> "Our first model is the 'Vanilla GAN'. This is the standard, original architecture utilizing Multi-Layer Perceptrons, or fully-connected layers. The Generator starts with 100 random noise values and passes them through a series of Linear layers, utilizing LeakyReLU activations and Batch Normalization. Eventually, it outputs a flat array of 784 pixels, which we reshape into a 28x28 image. The Discriminator does the exact reverse, flattening the image and processing it to guess 'real' or 'fake'. The main limitation here is that fully-connected layers treat images as just lists of numbers; they have no spatial awareness of what pixels are next to each other."

**Expected Questions & Answers:**
*   **Q:** *Why do you use LeakyReLU instead of standard ReLU?*
    *   **A:** Standard ReLU sets negative values to zero, which can "kill" gradients during backpropagation. LeakyReLU allows a small negative slope, keeping gradients flowing back to the Generator, preventing "dead" neurons.
*   **Q:** *Why does the Generator output use a Tanh activation function?*
    *   **A:** Tanh scales the output to the range $[-1, 1]$. We strictly normalize our real MNIST images to this exact same range using PyTorch transforms, ensuring the Generator outputs match the real data distribution perfectly.

---

## Slide 4: DCGAN & Conditional DCGAN (cDCGAN)
**Visuals/Content:**
*   **Diagram:** Flowchart of cDCGAN (Transposed convolutions fusing noise with class embeddings).
*   **Text:** Added `nn.Embedding` to fuse label data (0-9) directly into the generator and discriminator.
*   **Bullet:** True controllable synthesis (e.g., "force the generator to make a shoe").

**Speech:**
> "To solve the structural limitations of MLPs, we implemented a Deep Convolutional GAN, strictly following the Radford 2015 guidelines. But we didn't stop at random generation. We elevated the architecture by implementing a Conditional DCGAN (cDCGAN). In standard DCGANs, you have no control over what the network produces from the noise. By injecting an `nn.Embedding` layer and concatenating class labels directly with our spatial inputs in both the Generator and Discriminator, we essentially 'taught' the network what specific classes look like. This allows us to perform targeted synthesis—for instance, we can explicitly command the Generator to output a '7' or a 'Sneaker', giving us complete control over the latent space."

**Expected Questions & Answers:**
*   **Q:** *How exactly do you feed the label into the Discriminator?*
    *   **A:** We take the integer class label (0-9) and pass it through an Embedding layer to create a dense vector. We then reshape this vector into a 1x28x28 feature map, which is concatenated as an extra channel alongside the actual 1x28x28 image before passing it into the Discriminator's convolutional layers.
*   **Q:** *Why no pooling layers (like MaxPool) in DCGAN?*
    *   **A:** Max pooling is non-differentiable in a way that disrupts the GAN learning signal. Using strided convolutions allows the network to *learn* its own spatial downsampling method, keeping the entire pipeline fully differentiable.

---

## Slide 5: The Training Loop & Loss Functions
**Visuals/Content:**
*   **Equations:** Binary Cross-Entropy (BCE) Loss formula.
*   **Bullets:** 
    *   $Loss_D = BCE(D(real), 0.9) + BCE(D(fake), 0)$
    *   $Loss_G = BCE(D(fake), 1)$
*   **Note:** Label Smoothing = 0.9 (prevents overconfidence) and using `.detach()` in PyTorch.

**Speech:**
> "Training a GAN is mathematically framed as optimizing two Binary Cross-Entropy losses. The Discriminator is trained first; it wants to output a high probability for real images and 0 for fake images. Notably, we use 'Label Smoothing'—we tell the Discriminator that real images have a label of 0.9 rather than exactly 1.0. This prevents the Discriminator from becoming overly confident and halting gradient flow. Then we train the Generator. The Generator takes its fake images, feeds them to the Discriminator, and calculates its loss assuming the target label was 1.0. It's essentially saying: 'how badly did I fail to get a 1.0?' and uses that error to update its weights."

**Expected Questions & Answers:**
*   **Q:** *Why is Label Smoothing applied only to real labels and not fake labels?*
    *   **A:** We want the Discriminator to be absolutely sure that fakes are fakes (label 0). Smoothing the fake label could confuse the training. We smooth the real label to 0.9 so the Discriminator never reaches absolute certainty on real images, ensuring it always provides a usable gradient to the Generator.
*   **Q:** *Can you explain why you use `.detach()` on fake images when training the Discriminator?*
    *   **A:** If we don't detach, calculating the Discriminator's loss would build a computational graph all the way back through the Generator. When we call `.backward()`, PyTorch would incorrectly update the Generator's weights with the Discriminator's objective. Detaching safely cuts the graph.

---

## Slide 6: Evaluating the Models (FID Score & Its Limitations)
**Visuals/Content:**
*   **Title:** Fréchet Inception Distance (FID)
*   **Diagram/Box:** Real Images + Fake Images $\rightarrow$ Inception-v3 CNN $\rightarrow$ Feature Vectors $\mu, \Sigma$.
*   **Formula Image:** $FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g})$
*   **Warning Box:** FID is universally standard but biologically mismatched for grayscale 28x28 data.

**Speech:**
> "To evaluate our models quantitatively, we use the Fréchet Inception Distance, or FID score. We pass 10,000 real images and 10,000 generated images through a pre-trained Inception-v3 network to extract deep feature vectors, then calculate the mathematical distance between their distributions. Lower is better. However, as deep learning researchers, we must be critical of our metrics. Inception-v3 was trained on ImageNet—which consists of high-resolution, colored photographs of animals and objects. Applying it to 28x28 grayscale digits or clothes means the metric is biologically 'mismatched'. While it provides an excellent *relative* comparison to prove DCGAN beats Vanilla GAN, we must acknowledge that FID on MNIST is an imperfect absolute measure of image fidelity."

**Expected Questions & Answers:**
*   **Q:** *If FID is mismatched, why did you use it?*
    *   **A:** Because despite the domain shift of Inception-v3, the deep convolutional filters (like edge and texture detectors) still activate differently for sharp, coherent generated images versus blurry, noisy ones. Thus, it remains the most universally accepted heuristic for *comparing* multiple GAN architectures in the context of academic baselining.
*   **Q:** *What does an FID of zero mean?*
    *   **A:** An FID of exactly zero means the generated images have the identical mean and covariance as the real dataset in the feature space. In practice, this is impossible unless the model just perfectly memorized the training data and regurgitated it.

---

## Slide 7: Results & Targeted Synthesis
**Visuals/Content:**
*   **Left Side Images:** Side-by-side DCGAN vs Vanilla GAN (showing sharpness differences).
*   **Right Side Images:** A Conditional GAN grid showing exact control (e.g. forced rows of digits 0 through 9, or specific clothing types).
*   **Bottom Table:** FID Scores comparison across architectures.

**Speech:**
> "Looking at the visual results, the difference is striking. The Vanilla GAN produces recognizable but blurry images with significant topological noise. The DCGAN produces crisp, highly convincing samples. This is corroborated by our FID scores, where the convolutional models significantly outperformed the MLP models. More impressively, using our Conditional DCGAN, we successfully forced the network to generate highly targeted outputs on demand. On the screen, you can see a grid where we controlled the exact class of every single generated image, proving that the model hasn't just memorized outputs, but has actually mapped specific semantic meaning to its latent space embeddings."

**Expected Questions & Answers:**
*   **Q:** *Did the cDCGAN converge faster or slower than the standard DCGAN?*
    *   **A:** Conditioning a GAN often stabilizes training. By giving both the Generator and Discriminator explicitly labeled ground truth data, the Discriminator's job of structuring the feature space becomes more organized, which in turn usually helps the Generator avoid severe mode collapse.
*   **Q:** *Did you observe Mode Collapse in the Vanilla model?*
    *   **A:** Yes, the Vanilla GAN struggles significantly more with mode collapse, often repeatedly generating the same few digits (like 1s and 8s) because its lack of spatial awareness makes it harder to learn the diverse topological variants of more complex shapes like 2s or 5s.

---

## Slide 8: Ablation Study – Effect of Latent Dimension
**Visuals/Content:**
*   **Image:** `results/ablation/latent_dim_comparison.png` — a 3-panel figure generated by `ablation_latent.py`.
*   **Columns:** `z = 50` | `z = 100` | `z = 200`
*   **Caption:** Same architecture, same training — only the noise vector size changes.

**Speech:**
> "A common question in GAN research is: how does the size of the latent noise vector affect image quality? We ran a controlled ablation study using our trained DCGAN. We held all hyperparameters constant — same learning rate, same batch size, same number of epochs — and only varied the latent dimension z from 50 to 100 to 200. What you can see on the screen confirms the theoretical expectation: with z=50, the Generator is forced to compress too much information into too small a space, leading to limited diversity and mode-like repetition. At z=100, we get our best results — sharp, diverse outputs. At z=200, the search space actually becomes harder to optimize, and early training shows slightly less structured outputs. This analysis proves our default choice of z=100 is well-calibrated."

**Expected Questions & Answers:**
*   **Q:** *Why doesn't a bigger latent space always mean better results?*
    *   **A:** A larger latent space has more dimensions to search during gradient descent, which means the optimizer takes longer to find good structure. With a fixed training budget and learning rate, z=200 may not fully converge. It's a classic bias-variance tradeoff — you need enough capacity to represent diversity, but not so much that the optimization becomes intractable.
*   **Q:** *If you trained for 500 epochs instead of 100, would z=200 surpass z=100?*
    *   **A:** Very likely yes. Given sufficient training time, a larger latent space would encode more diverse features and likely produce richer outputs. Our 100-epoch budget was chosen to keep experiments comparable and computationally feasible.

---

## Slide 9: Class-wise Conditional Generation Analysis
**Visuals/Content:**
*   **Image:** `results/ablation/class_grid_fashion_mnist.png` — a 10×10 grid.
*   **Rows:** Each of the 10 Fashion-MNIST classes (T-shirt, Trouser, Sneaker etc.)
*   **Columns:** 10 different random noise vectors.
*   **Caption:** Same noise — only the class label changes. Shows the cDCGAN has truly separated class identity from style variation.

**Speech:**
> "This is the most powerful result we can show for our Conditional GAN. On this 10×10 grid, each row is a different class — T-shirts on the top row, all the way down to Ankle boots on the bottom. Each column uses the exact same random noise vector — the style seed. Notice how changing the class label completely changes the category of the generated item, while changing the noise column changes the style within that class. This proves our cDCGAN has genuinely disentangled 'what to generate' from 'how to generate it'. The class label controls semantic category, and the noise vector controls intra-class variation. This is exactly the kind of structured latent space that makes conditional GANs useful in real-world applications like data augmentation and controlled synthetic data generation."

**Expected Questions & Answers:**
*   **Q:** *Can you generate a Fashion-MNIST image of a specific Sneaker in a specific style?*
    *   **A:** Exactly. You pick your class label (7 = Sneaker) and your noise vector. With the same noise vector but different labels, you get different items in the same 'style'. This is a foundational property of conditional generation.
*   **Q:** *How is this different from just training 10 separate GANs?*
    *   **A:** Training 10 separate GANs would give you 10 isolated models with no shared understanding. Our single cDCGAN learns a shared feature space where visual concepts like edges, textures, and shapes are reused across classes. It's more efficient, more generalizable, and demonstrates the power of conditioning over isolation.

---

## Slide 10: Conclusion
**Visuals/Content:**
*   **Bullet Points:**
    *   Successfully implemented both architectures from scratch.
    *   Automated pipeline across multiple datasets: MNIST and Fashion-MNIST.
    *   Convolutional architectures (DCGAN) vastly outperform MLPs for image topology.
    *   Successfully implemented Conditional Generation (cDCGAN) for semantic control.
*   **Final Thanks:** Thank you for listening. Questions?

**Speech:**
> "In conclusion, we built and trained an end-to-end generative pipeline. We moved beyond a trivial baseline by extending our experiments across multiple distinct datasets, implementing advanced conditional synthesis via semantic embeddings, and critically analyzing our evaluation metrics. We proved that convolutional architectures fundamentally capture physical topologies far better than flat perceptrons, and demonstrated exact control over generated outputs. Thank you for your time, and we'd be happy to answer any further questions!"

**(End of Presentation)**
