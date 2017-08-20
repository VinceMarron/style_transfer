# StyleTransfer

Style Transfer as Optimal Transport

Demonstrates how to perform style transfer with l2-Wasserstein distance as loss function

Let me know if you would be interested in more explanation/theory behind this formulation.

Files:
vgg.py - unpacks 'imagenet-vgg-verydeep-19.mat' author: https://github.com/anishathalye

synthesize.py - synthesizes an image by transferring a 'style' onto a 'subject' image

example.ipynb - demonstrates usecase

kanagawa.jpg, surf.jpg - inputs used in example.ipynb

kanagawa_surf.png, kanagawa_surf2.png, kanagawa_surf.avi - outputs from example.ipynb

Requires: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat 


