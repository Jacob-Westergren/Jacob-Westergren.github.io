## Portfolio

---

### Group Projects

---

### Class-Conditioned Image Generation
<p>
Developed a Denoising Diffusion Probabilistic Model with a U-Net architecture utilizing self- and cross-attention for class-conditional image generation. The model was progressively tested on the MNIST, CIFAR-10, and LFWPeople-CelebA datasets, tackling increasingly challenging tasks. While it generated high-quality images and achieved impressive FID scores, further training is needed to reach state-of-the-art performance.
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
  Built a Vector Quantized Variational Auto-Encoder (VQ-VAE) to encode sign language gestures into latent vectors for reconstruction, aiming to investigate the potential building blocks of sign language. The model effectively broke down gestures into discrete components, but it struggled to capture subtle finger movements, particularly when the hands were positioned in ways that hindered camera detection of keypoints. 
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
  Implemented a model to generate speech from visual data, focusing on lip movements. The architecture consisted of a front-end with convolutional layers for feature extraction, conformers to process these features, and Whisper for ASR. Due to the model's large size, we had to cut training short because of limited compute power, resulting in suboptimal performance. However, the model showed clear progress in generating corresponding sounds from visual input, and with further training, it is expected to produce high-quality speech.
  
</p>
<ul>
  <li><b>Dataset:</b> GRID dataset</li>
  <li><b>Architecture:</b> Convolutional networks, conformers, Whisper model integration</li>
  <li><b>Results:</b> Enhanced mel-spectrogram quality with Whisper module</li>
</ul>
<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/Lip_to_Speech_arch.png?raw=true" alt="Lip to Speech model architecture" width="400" style="max-height: 400px;"/>
    <figcaption>Illustration of the Lip-to-Speech model architecture, where the Visual Front-End and Conformer make up the encoder and the speech synthesizer is the decoder.</figcaption>
  </figure>
</div>

<div style="text-align:center;">
  <figure style="display: inline-block; text-align: center;">
    <img src="images/Lip_to_Speech_mel.png?raw=true" alt="True vs Generated Mel-Spectrograms" width="400" style="max-height:500px"/>
    <figcaption>Comparison of the true and the generated mel-spectrogram.</figcaption>
  </figure>
</div>



---

## Bachelor Thesis

# <a href="https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=3&af=%5B%5D&searchType=SIMPLE&sortOrder2=title_sort_asc&query=quantum+support+vector+machine&language=en&pid=diva2%3A1779801&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-2225" target="_blank">Quantum Support Vector Machine</a>

<p> 
  In my Bachelor thesis, I compared the performance of quantum and classical kernels for Support Vector Machines (SVM) in binary classification tasks. 
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

## Research Project
At Östra High School, Alexander Hollmark and I classified the exoplanet HAT-P-30 using Ph.D. Tahir Yaqoob’s classification system. This project, done in collaboration with the Astronomical Society Tycho Brahe and advised by Simon Eriksson from the House of Science, involved a transit observation. We analyzed the brightness drop as the exoplanet passed between its star and our telescope. Due to weather conditions that arose during the observation, we supplemented our data with values from previous experiments. Our findings indicated that HAT-P-30 was a hot or super Jupiter, consistent with prior research.

<div style="text-align:center;"> 
  <figure style="display: inline-block; text-align: center;"> 
    <img src="images/exoplanet_classification_pic.png?raw=true" alt="Exoplanet Classification Experiment" width="400" style="max-height: 500px;"/>
    <figcaption> Image taken of me during the transit observation showing the experiment setup. </figcaption> 
  </figure> 
</div>
