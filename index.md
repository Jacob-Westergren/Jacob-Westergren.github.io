## Portfolio

---

### Projects

### Class-Conditioned Diffusion Model
<p>Implemented a U-Net architecture with self- and cross-attention for class-conditional image generation. Tested on MNIST, CIFAR-10, and LFWPeople-CelebA datasets, the model achieved high FID scores but requires further training to reach state-of-the-art performance.</p>
<img src="images/people_img.png?raw=true" alt="Generated images from the diffusion model"/>
<img src="images/DDPM_final_arch.png?raw=true" alt="U-Net architecture of diffusion model"/>

---

### Lip-to-Speech
<p>Developed a model that generates speech from visual lip movements and text using convolutional networks and conformers. The model also incorporated a pre-trained Whisper module in it's loss function, enhancing the quality of the generated mel-spectrograms. </p>
<img src="images/Lip_to_Speech_arch.png?raw=true" alt="Lip to Speech model architecture"/>

---

### Sign Language Reconstruction
<p>Built a Vector Quantized Variational Auto-Encoder (VQ-VAE) to encode sign language gestures into latent vectors for reconstruction, with the goal being to investigate the potential existence of building blocks in sign language. The model showed effective clustering of gestures and successful reconstruction but  struggled to capture nuanced finger movements. </p>
<img src="images/sign_language_rec.png?raw=true" alt="Sign language reconstruction visualization"/>

---
