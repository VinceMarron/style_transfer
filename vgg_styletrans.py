# Copyright (c) 2017 Vincent Marron | Released under MIT License

import tensorflow as tf
import numpy as np
from PIL import Image
import time


class TransferStyle(object):
  """Modifies a 'subject_image' to exhibit the visual style of a 'style_image'
     
    vgg19 weights binary (115.5 MB) can be downloaded from:         
    https://app.box.com/v/vgg19-conv-npy    
    md5sum bf8a930fec201a0a2ade13d3f7274d0e
    
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
        eval_out (bool): wether to open tf session and eval style description to np array
        pool_type (str): 'avg', 'max', or 'none', type of pooling to use
        last_layer (str): vgg network will process image up to this layer

    """
    with self.graph.as_default():

      self.style_desc = {}
      self.style_arr = tf.constant((np.expand_dims(style_image,0)[:,:,:,:3])
                                   .astype('float32'))

      x = self.style_arr-self.mean_pixel

      self.stop = self.all_layers.index(last_layer)+1

      for i, layer in enumerate(self.all_layers[:self.stop]):

        if layer[:2] == 're': x = tf.nn.relu(x)

        elif layer[:2] == 'po': x = self.pool_func(x, pool_type)

        elif layer[:2] == 'co':
          kernel = self.vgg_ph[layer+'_kernel']
          bias = self.vgg_ph[layer+'_bias']

          x = tf.nn.bias_add(tf.nn.conv2d(x, kernel,
                                          strides=(1, 1, 1, 1),
                                          padding='SAME'),bias)

          layer_shape = tf.shape(x, out_type=tf.int32)

          #flattens image tensor to (#pixels x #channels) assumes batch=1
          #treats each pixel as an observation of Gaussian random vector
          #in R^(#channels) and infers parameters
          stl_activs = tf.reshape(x, [layer_shape[1]*layer_shape[2], layer_shape[3]])
          mean_stl_activs = tf.reduce_mean(stl_activs, axis=0, keep_dims=True)
          covar_stl_activs = (tf.matmul(stl_activs - mean_stl_activs,
                                        stl_activs - mean_stl_activs, transpose_a=True)/
                              tf.cast(layer_shape[1]*layer_shape[2], tf.float32))

          #takes root of covar_stl_activs
          #(necessary for wdist, as tf cannot take eig of non-symmetric matrices)
          eigvals,eigvects = tf.self_adjoint_eig(covar_stl_activs)
          eigval_mat = tf.diag(tf.sqrt(tf.maximum(eigvals,0.)))
          root_covar_stl_activs = tf.matmul(tf.matmul(eigvects, eigval_mat)
                                                ,eigvects,transpose_b=True)

          trace_covar_stl = tf.reduce_sum(tf.maximum(eigvals,0))
          
          self.style_desc[layer] = (mean_stl_activs,
                                    trace_covar_stl,
                                    root_covar_stl_activs)
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

      if loss_layers==[]: loss_layers = self.all_layers[:self.stop]

      y = self.synth_arr-self.mean_pixel

      for i, layer in enumerate(self.all_layers[:self.stop]):

        if layer[:2] == 're': y = tf.nn.relu(y)

        elif layer[:2] == 'po': y= self.pool_func(y, pool_type)

        elif layer[:2] == 'co':
          kernel = self.vgg_ph[layer+'_kernel']
          bias = self.vgg_ph[layer+'_bias']

          y = tf.nn.bias_add(tf.nn.conv2d(y, kernel,
                                          strides=(1, 1, 1, 1),
                                      padding='SAME'),bias)
          if layer in loss_layers:
            layer_shape = tf.shape(y, out_type=tf.int32)

            #flattens image to (#pixels x #channels) assumes batch=1
            #treats each pixel as an observation of Gaussian random vector
            #in R^(#channels) and infers parameters

            synth_activs = tf.reshape(y, [layer_shape[1]*layer_shape[2], layer_shape[3]])
            mean_synth_activs = tf.reduce_mean(synth_activs, axis=0, keep_dims=True)
            covar_synth_activs = (tf.matmul(synth_activs - mean_synth_activs,
                              synth_activs - mean_synth_activs, transpose_a=True)/
                              tf.cast(layer_shape[1]*layer_shape[2], tf.float32))
            
            trace_covar_synth = tf.trace(covar_synth_activs)
            
            mean_stl_activs, trace_covar_stl, root_covar_stl_activs = self.style_desc[layer]
            
            mean_diff_squared = tf.reduce_sum(tf.square(mean_stl_activs-mean_synth_activs))

            covar_prod = tf.matmul(tf.matmul(root_covar_stl_activs,covar_synth_activs),root_covar_stl_activs)
            #trace of sqrt of matrix is sum of sqrts of eigenvalues
            var_overlap = tf.reduce_sum(tf.sqrt(tf.maximum(
                          tf.self_adjoint_eig(covar_prod)[0],0.)))

            #loss can be slightly negative because of the 'maximum' on eigvals of covar_prod
            #could fix with trace_covar_synth = tf.reduce_sum(tf.maximum(eigvals_synth,0))
            #but that would mean non-critical computation

            self.loss += mean_diff_squared+trace_covar_stl+trace_covar_synth-2*var_overlap



  def synthesize_image(self, savename, optimizer = 'adam', steps=50, lr=1., log_ims=True, report_int=10,
                      blur_int=0, blur_dim=15, blur_sig_sq=1.):
    """invokes an optimizer and creates synthesized image

    'savename' (str): ending in PIL image format (.jpg, .png, etc.)
    'optimizer' (str): 'adam', 'rmsprop' or 'bfgs' ('bfgs' runs through scipy interface,
      cannot log/blur images or report status for this option & 'lr' is ignored)
    'steps': (int>0) number of iterations
    'lr': (float) the learning rate for adam/rmsprop
    'log_ims' (boolean): if 'True' stores a snapshot of image being created at each step
    'report_int' (int>0): interval of steps at which time and loss will be printed
    'blur_int' (int, ignored if 0): interval at which to apply Gaussian blur to synth_image.
    'blur_dim' (int>0, preferably odd): pixel width & height of Gaussian kernel
    'blur_sig_sq' (float): sigma squared for Gaussian/RBF kernel

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
          
          if blur_int>0 and (step+1)%blur_int==0:
            sess.run(tf.assign(self.synth_arr, 
                               self.blur_func(self.synth_arr, blur_dim, blur_sig_sq)
                               , use_locking=True))
          
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

      print("synthesize image saved:", savename)

  @staticmethod    
  def pool_func(x, pool_type):
    """ runs 2x2 pooling function on spatial dimensions of x

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

  @staticmethod
  def blur_func(x, kerneldim=15, sigma_sq=1.):
    """ Convolves spatial dimenstions of x with gaussian rbf kernel

    Args:
        x (tensor): 4-d (batch x height x width x channels)
        kerneldim (int>0): preferably odd so kernel centered
        sigma_sq (float): parameter for rbf

    """

    tensor_shape = tf.shape(x)
    d=tf.range(kerneldim, dtype=tf.float32)
    d=tf.expand_dims((d-d[kerneldim//2])**2,0)
    dist=d+tf.transpose(d)
    kernel=tf.exp(-dist*(1./(2.*sigma_sq)))
    kernel = kernel/tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, (kerneldim, kerneldim,1,1))
    kernel = tf.tile(kernel, (1,1,tensor_shape[3],1))

    return tf.nn.depthwise_conv2d(x, kernel, [1,1,1,1], padding='SAME')
