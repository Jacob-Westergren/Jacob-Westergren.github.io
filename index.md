## Portfolio

---

### Projects

---

### Class-Conditioned Diffusion Model
<p>
  Implemented a U-Net architecture with self- and cross-attention for class-conditional image generation. The model was tested on MNIST, CIFAR-10, and LFWPeople-CelebA datasets. While it achieved high FID scores, further training is needed to reach state-of-the-art performance.
</p>
<ul>
  <li><b>Dataset:</b> MNIST, CIFAR-10, LFWPeople-CelebA</li>
  <li><b>Architecture:</b> U-Net with self- and cross-attention</li>
  <li><b>Results:</b> High FID scores, further training required</li>
</ul>
<div style="text-align:center;">
  <img src="images/people_img.png?raw=true" alt="Generated images from the diffusion model"/>
  <img src="images/DDPM_final_arch.png?raw=true" alt="U-Net architecture of diffusion model"/>
</div>

---

### Lip-to-Speech
<p>
  Developed a model that generates speech from visual lip movements and text using convolutional networks and conformers. By incorporating a pre-trained Whisper module into the loss function, the quality of generated mel-spectrograms was significantly improved.
</p>
<ul>
  <li><b>Dataset:</b> GRID dataset</li>
  <li><b>Architecture:</b> Convolutional networks, conformers, Whisper model integration</li>
  <li><b>Results:</b> Enhanced mel-spectrogram quality with Whisper module</li>
</ul>
<div style="text-align:center;">
  <img src="images/Lip_to_Speech_arch.png?raw=true" alt="Lip to Speech model architecture"/>
</div>

---

### Sign Language Reconstruction
<p>
  Built a Vector Quantized Variational Auto-Encoder (VQ-VAE) to encode sign language gestures into latent vectors for reconstruction. The goal was to explore potential building blocks in sign language. While the model effectively clustered gestures and reconstructed them, it struggled to capture subtle finger movements.
</p>
<ul>
  <li><b>Dataset:</b> RWTH Phoenix dataset</li>
  <li><b>Architecture:</b> VQ-VAE with latent codebook vectors</li>
  <li><b>Results:</b> Effective clustering, limited fine movement capture</li>
</ul>
<div style="text-align:center;">
  <img src="images/sign_language_rec.png?raw=true" alt="Sign language reconstruction visualization"/>
</div>

---
