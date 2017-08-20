# Style Transfer as Optimal Transport 

## Demonstrates how to perform style transfer with L2-Wasserstein distance as loss function. 

README work in progress..


1. A 'style' image is fed into the vgg network and the first two moments of the feature activations (means and covariances) are extracted as a representation of the style.
2. A 'subject' image is fed into the same network and similar statistics are extracted. 
3. The [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) between these parametrized distributions is used as a distance/loss function. 
4. Optimization is conducted to minimize this distance. 

Considering only the first two moments of the activations implicitly assumes the activations follow Gaussian distributions. This is not always true. 

Please let me know if you would be interested in more explanation/theory behind this formulation. If there is enough interest I will write it up formally. 

## Files:
example.ipynb - demonstrates usecase

vgg.py - unpacks 'imagenet-vgg-verydeep-19.mat' author: https://github.com/anishathalye

synthesize.py - synthesizes an image by transferring a 'style' onto a 'subject' image

kanagawa.jpg, surf.jpg - inputs used in example.ipynb

kanagawa_surf.png, kanagawa_surf2.png, kanagawa_surf.avi - outputs from example.ipynb

##Requires: 

must download http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat 
all testing done with tensorflow 1.3, python 3.5

README work in progress..
