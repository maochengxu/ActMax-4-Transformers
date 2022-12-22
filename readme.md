# CS-433 Project 2

## 1. Introduction
This project is part of the ML course at EPFL in collaboration with Ponce lab at Harvard university. Our main research question is:Do CNNs and Vision Transformers learn visual representations similar to those of the monkey brain?
Simulating visual neurons with Convolutional Neuron Networks have been of a great interest in order to comprehend the neuronal integration. Given an activation maximization pipeline generating so called prototypes (GAN- derived images obtained via neuronal responses), Vision Transformers are now investigated in order to testify their relevance for understanding similar integration processes. Transformers are thus able to develop super-stimulating prototypes. The prototypes converge quicker to maximum local values although these maxima are less significant in comparison with natural images. Eventually, Transformer's prototypes suggest a greater focus on shapes and on textures.

## 2. Descriptions

```
.
├── core                    # Set of useful classes for the whole pipeline provided by Ponce lab and that we adapted to Vits
│   ├── CNN_scorers.py      # pytorch CNN scorer that creates functions that map images to activation values of their hidden units.it was adapted to VITs
│   ├── GAN_utils.py        # torch version of fc6 GANsAlso provides loader and wrapper of various GANs.Currently we have * BigGAN
|   ├── Optimizers.py       # Optimizers developed by ponce lab
|   ├── geometry_utils.py   # Collection of some Geometric Utility function reusable across script
│   ├── insilico_exps.py    # runs the generations of images and saves the best one
│   ├── layer_hook_utils.py # functions to call in the forward and backward run
|   ├── montage_utils.py.   # class providd by ponce lab
│   └── robustCNN_utils.py  # load robust cnns
│
├── experiments             # contains the codes of all experiments
│   ├── ActMax_own_ViT.ipynb    # generate image from our own Vits implementation
│   ├── exp_for_CNN.ipynb       # get scores of cnn's generated image and natural images for tench and golf ball class
|   ├── generate_imgs.py        # generate images from Vit-b-16 model
|   ├── get_evolution_process.ipynb  # run evolutions for different models
│   ├── get_generate_scores.ipynb    # get the scores of generated images for each model
│   ├── get_nature_scores.ipynb # get the scores of natural images( from Imagenette dataset) for each model
|   ├── plots.ipynb       # violin plots and evolution plots ( all the models in the same plot)
│   └── similarity.ipynb  # Vits generated image interpretations
│
├── results                 # contains the results of all experiments 
├── transformers            # contains implementations of our Vits as wel as tf_exp.py which is the adaptatation of cnn_scorers to Vits
│   ├── tf_exp.py           # adaptatation of cnn_scorers to Vits
│   ├── model.pt            # our implementation of Vit saved
|   ├── vit.ipynb           # training our own vit
│   └── vit_model.py        # own vit implementations
└── Demo_ActMax.ipynb       # running vit-b-16 evolution
```

## 3. Requirements

- pytorch
- torchvision
- gc
- skimage
- cma == 3.0.3
- nevergrad == 0.4.2.post5

## 4. About us

| Name        | Sciper      | Email      |
| :---:       |    :----:   |  :---:     |
| Malek       | 288968      | malek.elmekki@epfl.ch|
| Maocheng    | 338251      | maocheng.xu@epfl.ch |
| Raphaël     | 123456      |            |
