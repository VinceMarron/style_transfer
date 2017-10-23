import os
from vgg_styletrans import TransferStyle
from argparse import ArgumentParser
from PIL import Image
import numpy as np

def main():
  parser = make_parser()
  args = parser.parse_args()
  
  if not os.path.isfile(args.vggnet):
    parser.error("Cannot find network weights file \n \
                  vgg19 weights binary (115.5 MB) can be downloaded from: \n \
                  https://app.box.com/v/vgg19-conv-npy \n \
                  md5sum: bf8a930fec201a0a2ade13d3f7274d0e")
  
  style_image = Image.open(args.style)
  subject_image = Image.open(args.subject)
  loss_layers = ['conv1_2', 'conv2_2','conv3_2', 
                 'conv3_4', 'conv4_2', 'conv4_4',]
  
  make=TransferStyle(args.vggnet)
  make.describe_style(style_image, last_layer='conv4_4')
  make.infer_loss(subject_image, loss_layers=loss_layers)
  make.synthesize_image(args.output, optimizer = 'bfgs', steps=args.steps)

def make_parser():
  parser = ArgumentParser()
  parser.add_argument('--vggnet', dest='vggnet', 
                      help='location of vgg19 weights binary', default='vgg_conv.npy')  
  parser.add_argument('--subject', dest='subject',
                      help='subject image, to be transformed', default='media/wave_small.jpg')
  parser.add_argument('--style', dest='style',
                      help='image portraying style to be transferred', default='media/wave_kngwa.jpg')
  parser.add_argument('--output', dest='output',
                      help='path for output', default='output.jpg')
  parser.add_argument('--steps', dest='steps', type=int,
                      help='# of steps optimizer can run', default=50)
  return parser


if __name__ == '__main__':
    main()