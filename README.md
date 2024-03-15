# Animation-GAN-InBetwener
This is an automatic In-Betweening tool for traditional 2D animation.
Given 2 drawings, this tool generates an intermediate one using a supervised autoencoder (U-Net like) and Generative Adversarial Networks (GAN). For an in-depth description of the model, experiments and results in its entirety, download the thesis document, [In-between frame generation for uncolored 2D animation](https://drive.google.com/file/d/1QYXvTovd8EeVhUQjnB7PIwIJXs7YR-JU/view?usp=sharing).

INSERT HORIMIYA GIF


**TL;DR**

Traditional 2D animation remains a largely manual process where each frame in a video is hand-drawn, as no robust algorithmic solutions exist to assist in this process. This project introduces a system that generates intermediate frames in an uncolored 2D animated video sequence using Generative Adversarial Networks (GAN), a relatively inexpensive deep learning approach widely used for tasks within the creative realm. We treat the task as a frame interpolation problem, and show that adding a GAN dynamic to a system significantly improves the perceptual fidelity of the generated images, as measured by perceptual metrics that aim to capture human judgment of image quality. Moreover, this thesis proposes a simple end-to-end training framework that avoids domain transferability issues that arise when leveraging components pre-trained on natural video. Lastly, we show that the two main challenges for frame interpolation in this domain, large motion and information sparsity, interact such that the magnitude of objects' motion across frames conditions the appearance of artifacts associated with information sparsity.


## Requirements
Description of the requirements (both the manual and automatic installation)

## Generate frames
To generate an in-between frame given a pair of end-frames, download the generator model's [checkpoints](https://drive.google.com/file/d/1HNBLPgWxvDbKNrPAua-SUQVr7d-zQgRl/view?usp=sharing), and save them in the `checkpoints/` directory. Then, run the following command:
```bash
python scripts/generate.py
```
This action will generate an intermediate frame for a provided sample pair of end-frames and save it in the `to_generate/wo_gt/` directory with the tile `in-between.png` along with a gif of the resulting triplet (sequence of 3 frames - the 2 end-frames + the in-between frame). To test your own pair of end-frames, replace the frames in the `to_generate/wo_gt/` directory with your own. 

### Frame triplets (with ground truth in-betweens)
To test the generation of a known in-between frame, run the following command:
```bash
python scrips/generate.py 'gt'
```
This action will generate an intermediate frame for a provided triplet and save it in the `to_generate/w_gt/` directory with the title `in-between.png` along with a gif of the resulting triplet (with the generated in-between), as well as a gif with the original triplet (with the real in-between). In addition, a gif of the resulting triplet, overlapping the generated frame with the real frame is also provided, in order to help visualize generation errors.


## Generate a video
To add generated frames to a video, run the following command:
```bash
python scripts/generate_from_vid.py
```
This action will produce a video with the same frame-rate as the input video, but replacing duplicate frames in the input video with generated frames (traditional 2D animation commonly produces videos with at least 50% duplicated frames to save costs). 


## Train system
To train the system, run the following command:
```bash
scripts/run_experiment.py --config experiment_configs/default.ini
```
This action will perform training from scratch with the parameters optimized by the authors of this repository by default. Feel free to modify the `experiment_configs/default.ini` file to modify the trainig parameters, as well as visualization and evaluation configurations. The `scripts/run_experiment.py` file contains commented explanations of what each parameter does. For example, the `pretrain` parameter allows you to start training with the model's fully trained weights, which is convenient for fine-tuning to your own dataset.


