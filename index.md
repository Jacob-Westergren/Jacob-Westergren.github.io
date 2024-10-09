## Portfolio

---

### Group Projects

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
  <figure style="display: inline-block; text-align: center;">
    <img src="images/people_img.png?raw=true" alt="Generated images from the diffusion model" width="300"/>
    <figcaption>Unconditionally generated images from the CelebA model pre-trained on LFWPeople, without EMA to the right and with EMA to the middle-right.</figcaption>
  </figure>
  
  <figure style="display: inline-block; text-align: center;">
    <img src="images/DDPM_final_arch.png?raw=true" alt="U-Net architecture of diffusion model" width="300"/>
    <figcaption>Simplified schematic of the model, where the different coloured areas showcase the encoder, middle block, and decoder. The black arrows represent the variable X’s path, the white arrows indicate the U-net’s residual connections, and the dotted arrows show the time and class conditioning variables’ paths.</figcaption>
  </figure>
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
  <figure style="display: inline-block; text-align: center;">
    <img src="images/Lip_to_Speech_arch.png?raw=true" alt="Lip to Speech model architecture" width="300"/>
    <figcaption>Illustration of the Lip-to-Speech model architecture, where the Visual Front-End and Conformer make up the encoder and the speech synthesizer is the decoder.</figcaption>
  </figure>
</div>

---

### Sign Language Reconstruction
<p>
  Built a Vector Quantized Variational Auto-Encoder (VQ-VAE) to encode sign language gestures into latent vectors for reconstruction, with the goal of exploring the potential existence and usage of building blocks in sign language. While the model effectively broke down gestures into building blocks that it could reconstruct from, it struggled to capture subtle finger movements, particularly when the hands were positioned in such a way that most keypoints were undetectable.
</p>
<ul>
  <li><b>Dataset:</b> RWTH Phoenix dataset</li>
  <li><b>Architecture:</b> VQ-VAE with latent codebook vectors</li>
  <li><b>Results:</b> Effective clustering, limited fine movement capture</li>
</ul>
<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/sign_language_rec_gif.gif?raw=true" alt="Sign language reconstruction visualization" width="300"/>
    <figcaption>Sign language reconstruction visualization in motion.</figcaption>
  </figure>
</div>

---

## Bachelor Thesis

# Quantum Support Vector Machine
<p> 
  For my Bachelor thesis paper, I compared the performance of quantum and classical kernels for Support Vector Machines (SVM) in binary classification tasks, . 
  The quantum computations were simulted using the Qiskit library for their ease of use, and our results showcased a much stronger performance when utilizing a quantum kernel compared to the classical Radial Basis Function (RBF) kernel for our training data. 
</p>

<ul> 
  <li><b>Dataset:</b> Toy-generated dataset</li> 
  <li><b>Architecture:</b> Support Vector Machines with quantum and RBF kernels</li> 
  <li><b>Results:</b> Quantum kernel outperformed classical kernel</li> </ul> 

<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/QSVM.png?raw=true" alt="Quantum Support Vector Machine Flowchart" width="300"/>
    <figcaption> A flowchart of the used pipeline, divided into three modules: kernel training, SVM training and finally prediction.</figcaption>
  </figure>
</div>

