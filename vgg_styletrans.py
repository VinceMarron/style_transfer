# Copyright (c) 2018 Vincent Marron | Released under MIT License

import tensorflow as tf
import numpy as np
from PIL import Image
import time

class TransferStyle(object):
  """Modifies a 'subject_image' to exhibit the visual style of a 'style_image'
     
    vgg19 weights binary (115.5 MB) can be downloaded from:         
    https://app.box.com/v/vgg19-conv-npy    
    md5sum: bf8a930fec201a0a2ade13d3f7274d0e
    
    Args:
        vgg_weight_path (str): Path to the '.npy' binary containing vgg19 weights

    """
  def __init__(self, vgg_weight_path):

    self.all_layers = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4']

    self.graph = tf.Graph()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
                                allow_growth=True)
    self.config = tf.ConfigProto(gpu_options=gpu_options,
                                 operation_timeout_in_ms=99999)

    vgg_npy = np.load(vgg_weight_path).item()
    self.feed_dict = {param+':0': vgg_npy[param] for param in vgg_npy.keys()}

    self.vgg_ph = {}
    with self.graph.as_default():
      self.mean_pixel = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32)
      
      for param in vgg_npy.keys():
        self.vgg_ph[param] = tf.placeholder(tf.float32, vgg_npy[param].shape, param)
        

  def describe_style(self, style_image, eval_out=False, pool_type='avg', last_layer='conv5_4'):
    """ Runs the 'style_image' through the vgg network and extracts a statistical
    description of the activations at convolution layers

    Args:
        style_image (PIL image object): displays the style to be transferred
        eval_out (bool): whether to open tf session and eval style description to np array
        pool_type (str): 'avg', 'max', or 'none', type of pooling to use
        last_layer (str): vgg network will process image up to this layer

    """
    with self.graph.as_default():

      self.style_desc = {}
      self.style_arr = tf.constant((np.expand_dims(style_image,0)[:,:,:,:3])
                                   .astype('float32'))

      x = self.style_arr-self.mean_pixel
      
      for layer in self.all_layers[:self.all_layers.index(last_layer)+1]:
          
        if layer[:4] == 'relu': x = tf.nn.relu(x)

        elif layer[:4] == 'pool': x = pool_func(x, pool_type)

        elif layer[:4] == 'conv':
          kernel = self.vgg_ph[layer+'_kernel']
          bias = self.vgg_ph[layer+'_bias']

          x = tf.nn.bias_add(tf.nn.conv2d(x, kernel,
                                          strides=(1, 1, 1, 1),
                                          padding='SAME'),bias)

          mean, cov = calc_2_moments(x)

          #takes root of covar_stl_activs
          #(necessary for later step, as tf cannot take eig of non-symmetric matrices)
          eigvals,eigvects = tf.self_adjoint_eig(cov)
          eigroot_mat = tf.diag(tf.sqrt(tf.maximum(eigvals,0.)))
          root_cov = tf.matmul(tf.matmul(eigvects, eigroot_mat)
                                                ,eigvects,transpose_b=True)

          tr_cov = tf.reduce_sum(tf.maximum(eigvals,0))
          
          self.style_desc[layer] = (mean,
                                  tr_cov,
                                  root_cov)
      if eval_out==True:
        with tf.Session(graph=self.graph, config=self.config) as sess:
          self.style_desc = sess.run(self.style_desc, feed_dict=self.feed_dict)


  def infer_loss(self, subj_image, loss_layers=[], pool_type='avg'):
    """ Runs the 'subj_image' through the vgg network, extracts a statistical
    description of the activations at convolution layers and compares this 
    with that of the 'style_image'

    Args:
        subj_image (PIL image object): image onto which style should be transferred
        loss_layers (list of strs): layers to include in loss. if blank uses all conv layers 
        pool_type (str): 'avg', 'max', or 'none', type of pooling to use

    """
    with self.graph.as_default():
      self.loss =0
      
      self.synth_arr = tf.Variable((np.expand_dims(subj_image,0)[:,:,:,:3]).astype('float32')
                                   , trainable=True, dtype=tf.float32)

      if loss_layers==[]: loss_layers = self.all_layers
        
      y = self.synth_arr-self.mean_pixel

      for layer in self.all_layers:
        if layer[:4] == 'relu': y = tf.nn.relu(y)

        elif layer[:4] == 'pool': y= pool_func(y, pool_type)

        elif layer[:4] == 'conv':
          if layer not in self.style_desc.keys(): break
          
          kernel = self.vgg_ph[layer+'_kernel']
          bias = self.vgg_ph[layer+'_bias']

          y = tf.nn.bias_add(tf.nn.conv2d(y, kernel,
                                          strides=(1, 1, 1, 1),
                                      padding='SAME'),bias)
          if layer in loss_layers:
            mean_synth, cov_synth = calc_2_moments(y)
            
            dist = calc_l2wass_dist(self.style_desc[layer], mean_synth, cov_synth)
            
            self.loss += dist
            

  def synthesize_image(self, savename, optimizer = 'adam', steps=50, lr=1., 
                       log_ims=True, report_int=10,):
    """invokes an optimizer and creates synthesized image

    'savename' (str): ending in PIL image format (.jpg, .png, etc.)
    'optimizer' (str): 'adam', 'rmsprop' or 'bfgs' ('bfgs' runs through scipy,
       'lr' and 'report_int' are ignored)
    'steps': (int>0) number of iterations
    'lr': (float) the learning rate for adam/rmsprop
    'log_ims' (boolean): if 'True' snapshots image being created at each step
    'report_int' (int>0): interval of steps at which time and loss will be printed

    """

    start = time.time()

    with tf.Session(graph=self.graph, config=self.config) as sess:
      self.imagelist = []

      if optimizer == 'bfgs':
        sess.run(tf.global_variables_initializer())
        
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                   options={'maxiter': steps},
                                   var_list = [self.synth_arr],
                                   var_to_bounds={self.synth_arr: (0.,255.)})
        
        print("loss:","{:,.2f}".format(sess.run(self.loss, feed_dict=self.feed_dict)))
        
        if log_ims:
          def logimage(arr):
            self.imagelist.append(np.clip(arr[0], 0, 255).astype('uint8'))
        
          optimizer.minimize(session=sess, fetches=[self.synth_arr], 
                           feed_dict=self.feed_dict, loss_callback=logimage)
        
        else:
          optimizer.minimize(session=sess, feed_dict=self.feed_dict)
          
        

      else:
        if optimizer == 'rmsprop':
          self.train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
                             self.loss, var_list = [self.synth_arr])

        else:
          self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                             self.loss, var_list = [self.synth_arr])

        sess.run(tf.global_variables_initializer())

        print("loss:","{:,.2f}".format(sess.run(self.loss, feed_dict=self.feed_dict)))

        for step in range(steps):
                   
          sess.run(self.train_op, feed_dict=self.feed_dict)

          if log_ims:
            self.imagelist.append(np.clip(self.synth_arr[0].eval(), 0, 255).astype('uint8'))

          if (step+1)%report_int==0:
            print("step:", step+1,
                "time:", "{:4.1f}".format(time.time()-start),
                "loss:", "{:,.2f}".format(sess.run(self.loss, feed_dict=self.feed_dict)))
            
      sess.run(tf.assign(self.synth_arr, 
                         tf.clip_by_value(self.synth_arr, 0, 255), 
                         use_locking=True))

      print("ALL DONE \n" + "post clip | time:", "{:4.1f}".format(time.time()-start),
            "final loss:","{:,.2f}".format(sess.run(self.loss, feed_dict=self.feed_dict)))

      self.img_out = Image.fromarray(self.synth_arr[0].eval().astype('uint8'))
      self.img_out.save(savename)

      print("synthesized image saved:", savename)

  
def pool_func(x, pool_type):
  """Runs 2x2 pooling function on spatial dimensions of x

  Args:
      x (tensor): 4-d (batch x height x width x channels)
      pool_type (str): 'avg', 'max', or 'none'

  """
  if pool_type=='avg':
    out = tf.nn.avg_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
  elif pool_type=='max':
    out = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
  elif pool_type=='none': out = x

  return out



def calc_2_moments(tensor):
  """flattens tensor and calculates sample mean and covariance matrix 
  along last dim (presumably channels)"""
  
  shape = tf.shape(tensor, out_type=tf.int32)
  n = tf.reduce_prod(shape[:-1])
  
  flat_array = tf.reshape(tensor, (n, shape[-1]))
  mu = tf.reduce_mean(flat_array, axis=0, keepdims=True)
  cov = (tf.matmul(flat_array - mu,flat_array - mu, transpose_a=True)/
                    tf.cast(n, tf.float32))
  
  return mu, cov



def calc_l2wass_dist(layer_style_desc, mean_synth, cov_synth):
  """Calculates (squared) l2-Wasserstein distance between gaussians
  parameterized by first two moments of style and synth activations"""
  
  mean_stl, tr_cov_stl, root_cov_stl = layer_style_desc
  
  #tr_cov_synth = tf.trace(cov_synth)
  tr_cov_synth = tf.reduce_sum(tf.maximum(
                tf.self_adjoint_eig(cov_synth)[0],0.))
  
  
  mean_diff_squared = tf.reduce_sum(tf.square(mean_stl-mean_synth))

  cov_prod = tf.matmul(tf.matmul(root_cov_stl,cov_synth),root_cov_stl)
  
  #trace of sqrt of matrix is sum of sqrts of eigenvalues
  var_overlap = tf.reduce_sum(tf.sqrt(tf.maximum(
                tf.self_adjoint_eig(cov_prod)[0],0.1)))

  #loss can be slightly negative because of the 'maximum' on eigvals of cov_prod
  #could fix with  tr_cov_synth= tf.reduce_sum(tf.maximum(cov_synth,0))
  #but that would mean extra non-critical computation

  dist = mean_diff_squared+tr_cov_stl+tr_cov_synth-2*var_overlap
  
  ### above dist written out in latec:
  #\mathcal{W}_2(\mathcal{N}(\mu_{x},\Sigma_{x}),\mathcal{N}(\mu_{y},\Sigma_{y}))^2
  #&= \inf_{g \in G(\mathcal{N}^x,\mathcal{N}^y)} \mathbb{E}_{g}||x-y||^2 \\
  #&= ||\mu_x-\mu_y||^2 + \mbox{tr} (\Sigma_x)+ \mbox{tr} (\Sigma_y) 
  #- 2\mbox{tr} \left((\Sigma_y^{\frac{1}{2}}\Sigma_x\Sigma_y^{\frac{1}{2}})^{\frac{1}{2}}\right)
  
  
  return dist
