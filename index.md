## Portfolio

---

### Group Projects

---

### Class-Conditioned Image Generation
<p>
  Developed a Denoising Diffusion Probabilistic Model with a U-Net architecture using self- and cross-attention for class-conditional image generation. The model was tested on MNIST, CIFAR-10, and LFWPeople-CelebA datasets. While it achieved high FID scores, further training is needed to reach state-of-the-art performance.
</p>
<ul>
  <li><b>Dataset:</b> MNIST, CIFAR-10, LFWPeople-CelebA</li>
  <li><b>Architecture:</b> U-Net with self- and cross-attention</li>
  <li><b>Results:</b> High FID scores, further training required</li>
</ul>
<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/people_img.png?raw=true" alt="Generated images from the diffusion model" width="400"/>
    <figcaption>Unconditionally generated images from the CelebA model pre-trained on LFWPeople, without EMA to the right and with EMA to the middle-right.</figcaption>
  </figure>
  
  <figure style="display: inline-block; text-align: center;">
    <img src="images/DDPM_final_arch.png?raw=true" alt="U-Net architecture of diffusion model" width="400"/>
    <figcaption>Simplified schematic of the model, where the different coloured areas showcase the encoder, middle block, and decoder. The black arrows represent the variable X’s path, the white arrows indicate the U-net’s residual connections, and the dotted arrows show the time and class conditioning variables’ paths.</figcaption>
  </figure>
</div>

---

### Sign Language Reconstruction
<p>
  Built a Vector Quantized Variational Auto-Encoder (VQ-VAE) to encode sign language gestures into latent vectors for reconstruction, aiming to investigate the potential building blocks of sign language. While the model effectively broke down gestures into blocks, these components lacked the subtle finger movements of the original gestures, especially when the hands were positioned in such a way that the camera couldn't detect most keypoints. 
</p>
<ul>
  <li><b>Dataset:</b> RWTH Phoenix dataset</li>
  <li><b>Architecture:</b> VQ-VAE with latent codebook vectors</li>
  <li><b>Results:</b> Effective clustering, limited fine movement capture</li>
</ul>
<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/sign_language_rec_gif.gif?raw=true" alt="Sign language reconstruction visualization" width="400"/>
    <figcaption>Sign language reconstruction visualization in motion.</figcaption>
  </figure>
</div>

---

### Speech Generation from Lip Movements
<p>
  Implemented a model that generates speech from visual lip movements and text using convolutional networks and conformers. By incorporating a pre-trained Whisper module into the loss function, the quality of generated mel-spectrograms was significantly improved.
</p>
<ul>
  <li><b>Dataset:</b> GRID dataset</li>
  <li><b>Architecture:</b> Convolutional networks, conformers, Whisper model integration</li>
  <li><b>Results:</b> Enhanced mel-spectrogram quality with Whisper module</li>
</ul>
<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/Lip_to_Speech_arch.png?raw=true" alt="Lip to Speech model architecture" width="400" style="max-height: 500px;"/>
    <figcaption>Illustration of the Lip-to-Speech model architecture, where the Visual Front-End and Conformer make up the encoder and the speech synthesizer is the decoder.</figcaption>
  </figure>
</div>

---

## Bachelor Thesis

# <a href="https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=3&af=%5B%5D&searchType=SIMPLE&sortOrder2=title_sort_asc&query=quantum+support+vector+machine&language=en&pid=diva2%3A1779801&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-2225" target="_blank">Quantum Support Vector Machine</a>

<p> 
  For my Bachelor thesis, I compared the performance of quantum and classical kernels for Support Vector Machines (SVM) in binary classification tasks. 
  Using the Qiskit library for all simulations of quantum computations, we found that the quantum kernel significantly outperformed the classical Radial Basis Function (RBF) kernel on our training data, suggesting that quantum kernels could be a valuable asset for machine learning. 
</p>

<ul> 
  <li><b>Dataset:</b> Toy-generated dataset</li> 
  <li><b>Architecture:</b> Support Vector Machines with quantum and RBF kernels</li> 
  <li><b>Results:</b> Quantum kernel outperformed classical kernel</li> 
</ul> 

<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/QSVM.png?raw=true" alt="Quantum Support Vector Machine Flowchart" width="400" style="max-height: 500px;"/>
    <figcaption>A flowchart of the used pipeline, divided into three modules: kernel training, SVM training, and finally prediction.</figcaption>
  </figure>
</div>
