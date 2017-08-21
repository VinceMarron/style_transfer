# Style Transfer as Optimal Transport 

Currently a work in progress.. please stay tuned for a full README. 

## An algorithm that transfers the distribution of visual characteristics, or style, of one image onto a second image via an [Optimal Transport](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)) plan. Implemented in [Tensorflow](https://github.com/tensorflow/tensorflow).

1. A 'style' image is fed into the [vgg network](https://arxiv.org/pdf/1409.1556.pdf) which maps RGB-pixel values to an abstract feature space that increases in complexity with network depth (initially 64 features, eventually 512). The first two moments of the feature activations (means and covariances) at select layers are extracted as a representation of the style.
2. A 'subject' image is fed into the same network and similar statistics are extracted. 
3. The [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) between these parametrized distributions is used as a distance/loss function. 
4. Optimization is conducted to minimize this distance. 

*Considering only the first two moments of the activations implicitly assumes the activations follow Gaussian distributions. This is not always true. 

Please let me know if you would be interested in more explanation/theory behind this formulation. If there is enough interest I will write it up formally. 

## Files:
[example.ipynb](https://github.com/VinceMarron/StyleTransfer/blob/master/example.ipynb) - notebook that demonstrates use case

[vgg.py](https://github.com/VinceMarron/StyleTransfer/blob/master/vgg.py) - unpacks 'imagenet-vgg-verydeep-19.mat' author: https://github.com/anishathalye, citation below

[synthesize.py](https://github.com/VinceMarron/StyleTransfer/blob/master/synthesize.py) - synthesizes an image by transferring a 'style' onto a 'subject' image


##Requires: 

must download http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat 
all testing done with tensorflow 1.3, python 3.5

README work in progress..




@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {\url{https://github.com/anishathalye/neural-style}},
  note = {commit xxxxxxx}
}
