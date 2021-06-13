# Tutorial: TripletLoss for re-identification using FastAI v2

This tutorial demonstrates the use of the `TripleLoss` loss function in FastAI v2 (running over Pytorch). TripletLoss is one of the leading ways to match an image (or sound or other signal) with a large database of images, even if there are very few matches for that particular input. Joint-embedding networks (aka Twin or Siamese networks) are a common alternative for this "low k-shot" challenge. According to [that article I read](tk), triplet loss outperformed twin networks in all of the tk studies where both approaches were tried. Another option for re-identification is Scale Invariant Feature Transform (SIFT) which is implemented in OpenCV. I haven't seen any papers comparing SIFT and statistical ML techniques in real-world tasks. 

The tutorial walks you through:

- Creating the necessary conda environment
- Downloading and expanding the MNIST dataset of handwritten digits
- Creating and training a model using triplet loss
- Applying this model to a few images of a new glyph (a hand-drawn star)
- Finding the closest neighbors to the new images
- Visualizing that the model tightly clusters the new glyph
- Demonstrating that the model can be used for re-identification of the new glyph

MNIST is an easy problem for modern statistical ML, with relatively small data and a small input size. This makes it a good candidate for a tutorial. I will try to highlight the places where you'll need to alter parameters when faced with a more realistic task, like animal re-identification. 

The code assumes that you have a CUDA-accelerated GPU for training and inferencing. The tutorial runs quickly on a 2080 (about 30 seconds per epoch during training, about 20ms to inference) but runs _much_ slower on a mobile GPU like a GeForce GTX 1650 (30 minutes per epoch on my Surface Book 3). I assume that porting it to CPU wouldn't be difficult, but I think it would be quite slow. If you don't have a local GPU, I'd suggest using a GPU-powered cloud compute resource, such as Azure ML. (Let me know if you'd like to see a tutorial on running this in Azure ML.) 

## The problem of re-identification

Our photo libraries are filled with images of our friends and family. A machine learning model that focuses on "classification" can successfully tell us that we have lots of photos of "smiling man" or "smiling woman." Such models will not, generally, tell us which photos are of Uncle Al and Aunt Betty. _That_ task -- re-identifying individuals -- is a different problem and requires different approaches.

Beyond identifying Uncle Al and Aunt Betty in a photo library, re-identification is a common problem for wildlife biologists. Many species have photographic catalogs taken over years and decades, and many species have some distinctive features that can be used to identify individuals. 

### Some animals and how they may be re-identified

| Species | &nbsp | &nbsp |
| --- | --- | --- |
| Humpback Whales | tk | tk |
| Tigers | tk | tk | 
| Dolphins | tk | tk | 
| Manta Rays | tk | tk | 


## Create the Python environment 

I developed this tutorial on Ubuntu 20.04, Cuda 10.2, PyTorch 1.7, FastAI 2.3. I have to admit that the rigmarole of exactly recreating a GPU-enabled virtual environment is a little beyond me, but I _think_ you have to install CUDA manually and then when you run the `conda create` command below, I _think_ it will download properly-configured versions of the various libraries. 

Prequisites: 
1. [Install conda](tk) 
1. [Install CUDA](tk)

1. Clone this repo and change into the directory.
1. Create the environment:
    ```bash
    conda create -f 'conda_environment.yml'
    ```
1. Activate the environment for use with:
   ```bash
   conda activate fastai
   ```
1. Confirm the environment by running:
   ```bash
   python .\versions.py
   ```
Which should result in something similar to:
```bash
Pytorch : 1.7.0, FastAI : 2.3.1, CUDA? : True
```

The environment also installs Jupyter. You may need to restart your terminal session to have the `jupyter` in your path. Run:

```bash
jupyter notebook TripletLossTutorial.ipynb
```

and continue from there.