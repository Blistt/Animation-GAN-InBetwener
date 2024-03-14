# Animation-GAN-InBetwener
This is an automatic In-Betweening tool for traditional 2D animation.
Given 2 drawings, this tool generates an intermediate one using a supervised autoencoder and Generative Adversarial Networks (GAN). For an in-depth description of the model, experiments and results in its entirety, download the thesis document INSERT LINK. 

INSERT ITADORI IMAGE
TL;DR
Traditional 2D animation remains a largely manual process where each frame in a video is hand-drawn, as no robust algorithmic solutions exist to assist in this process. This project introduces a system that generates intermediate frames in an uncolored 2D animated video sequence using Generative Adversarial Networks (GAN), a relatively inexpensive deep learning approach widely used for tasks within the creative realm. We treat the task as a frame interpolation problem, and show that adding a GAN dynamic to a system significantly improves the perceptual fidelity of the generated images, as measured by perceptual oriented metrics that aim to capture human judgment of image quality. Moreover, this thesis proposes a simple end-to-end training framework that avoids domain transferability issues that arise when leveraging components pre-trained on natural video. Lastly, we show that the two main challenges for frame interpolation in this domain, large motion and information sparsity, interact such that the magnitude of objects' motion across frames conditions the appearance of artifacts associated with information sparsity.

INSERT HORIMIYA GIF

## Requirements
Description of the requirements (both the manual and automatic installation)

## Generate frames
Description of how to generate frames, with an image explaining this.

## Generate a video
Description of how to generate a video, along with a gif of a video (maybe?)

## Train system
Description of how to train a system
