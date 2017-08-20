import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import time
import vgg


class SynthesizeImage(object):
  """Synthesizes an image that portrays subj_image in the style of style_image.

  """ 
  def __init__(self, style_image, subj_image, path):

    self.graph = tf.Graph()
    self.graph.device('/gpu:0')
    self.path = path
    
    with self.graph.as_default():
      self.vgg_weights, self.mean_pixel = vgg.load_net(
                    self.path + 'imagenet-vgg-verydeep-19.mat')
    
      
            
      self.style_arr=tf.constant((np.expand_dims(style_image,0)[:,:,:,:3]
                                  -self.mean_pixel).astype('float32'))
      
      self.subj_arr = tf.constant((np.expand_dims(subj_image,0)[:,:,:,:3]
                                  -self.mean_pixel).astype('float32'))

      self.synth_arr = tf.Variable(self.subj_arr, trainable=True, dtype=tf.float32)

      self.loss = 0.

  def get_style_desc(self, layers):
    """ Runs the inputted image through the vgg network and extracts a statistical
    description of the activations
    """
    with self.graph.as_default():               
      self.style_desc = {}             
      stl_at_layers = vgg.net_preloaded(self.vgg_weights, self.style_arr, 'avg')

      for layer in layers:

        stl_layer_shape = tf.shape(stl_at_layers[layer][0], out_type=tf.int32)

        #flattens image to # pixels x # channels array
        stl_activs = tf.reshape(stl_at_layers[layer][0], 
                        [stl_layer_shape[0]*stl_layer_shape[1], stl_layer_shape[-1]])
        
        mean_stl_activs = tf.reduce_mean(stl_activs, axis=0, keep_dims=True)
        
        covar_stl_activs = (tf.matmul(stl_activs - mean_stl_activs, 
                            stl_activs - mean_stl_activs, transpose_a=True)
                            /tf.cast(stl_layer_shape[0]*stl_layer_shape[1], tf.float32))

        #performs eigendecomposition to take root of covar of stl activations

        eigvals,eigvects = tf.self_adjoint_eig(covar_stl_activs)
        eigval_mat = tf.diag(tf.sqrt(tf.maximum(eigvals,0)))  
        root_covar_stl_activs = tf.matmul(tf.matmul(eigvects, eigval_mat)
                                                ,eigvects,transpose_b=True)
                         
        self.style_desc[layer] = (mean_stl_activs, tf.trace(covar_stl_activs), root_covar_stl_activs)

  def infer_loss(self, loss_layers):
    """ Compares the staistical descriptions of the style and synthized images
    in order to calculate the extent to which they differ (the loss)
    
    """ 
    with self.graph.as_default():

      synth_at_layers = vgg.net_preloaded(self.vgg_weights, self.synth_arr, 'avg')

      for loss_layer in loss_layers:

        mean_stl_activs, var_stl, root_covar_stl_activs = self.style_desc[loss_layer]

        synth_layer_shape = tf.shape(synth_at_layers[loss_layer][0], out_type=tf.int32)

        #becomes (# pixels x # channels) 
        synth_activs = tf.reshape(synth_at_layers[loss_layer][0], 
                                [synth_layer_shape[0]*synth_layer_shape[1], synth_layer_shape[-1]])

        mean_synth_activs = tf.reduce_mean(synth_activs, axis=0, keep_dims=True)                
        covar_synth_activs = (tf.matmul(synth_activs - mean_synth_activs, 
                                synth_activs - mean_synth_activs, 
                                transpose_a=True)
                               /tf.cast(synth_layer_shape[0]*synth_layer_shape[1], tf.float32))
                
        squared_diff_means = tf.reduce_sum(tf.square(mean_stl_activs-mean_synth_activs))
        
        var_synth = tf.trace(covar_synth_activs)
        
        var_prod = tf.matmul(tf.matmul(root_covar_stl_activs,covar_synth_activs),root_covar_stl_activs)
        
        var_overlap = tf.reduce_sum(tf.sqrt(tf.maximum(
                        tf.self_adjoint_eig(var_prod)[0],0)))

        self.loss += squared_diff_means+var_stl+var_synth-2*var_overlap

  def scipy_optimizer(self, savename, maxiters):    
    """invokes the l-bfgs optimizer to create a synthesized image"""

    start = time.time()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)

    with tf.Session(graph=self.graph, config=tf.ConfigProto(
                    gpu_options=gpu_options, operation_timeout_in_ms=15000)) as sess:    

      sess.run(tf.global_variables_initializer())

      print("loss:","{:,}".format(sess.run(self.loss)))

      optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                         options={'maxiter': maxiters}, 
                                                         var_list = [self.synth_arr])

      optimizer.minimize(session=sess)
      print("loss:","{:,}".format(sess.run(self.loss)))


      print("loss pre-clip:","{:,}".format(sess.run(self.loss)))
      sess.run(self.synth_arr.assign(tf.clip_by_value(
            self.synth_arr + self.mean_pixel, 0, 255)-self.mean_pixel))
      print("time:", "{:4.1f}".format(time.time()-start),
            "loss post-clip:","{:,}".format(sess.run(self.loss)))
      
      self.img_out = (self.synth_arr[0].eval()+self.mean_pixel).astype('uint8')
      
      plt.figure(figsize=(20,40))
      plt.imshow(self.img_out)

      imsave((self.path + savename + '.png'), self.img_out)

      ##Maybe want to set subj_arr to synth_arr but maybe not


  def build_image(self, savename, steps, lr, report=10, log_ims = True):
    """invokes the adam optimizer native to tensorflow to create a synthesized image"""

    start = time.time()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)

    with tf.Session(graph=self.graph, 
                    config=tf.ConfigProto(gpu_options=gpu_options, operation_timeout_in_ms=150000)) as sess:
      
      self.train_op = tf.train.AdamOptimizer(learning_rate=lr, use_locking=True).minimize(
                             self.loss, var_list = [self.synth_arr])
      
      sess.run(tf.global_variables_initializer())

      if log_ims:
        self.imagelist = []
        self.imagelist.append(self.synth_arr[0].eval()+self.mean_pixel)

      print("loss:","{:,}".format(sess.run(self.loss)))


      for step in range(steps):                  
        sess.run(self.train_op)
        
        if (step+1)%report==0:
          print("step:", step+1,
                "time:", "{:4.1f}".format(time.time()-start),
                "loss:", "{:,}".format(sess.run(self.loss)))

        if log_ims:    
          self.imagelist.append(np.clip(self.synth_arr[0].eval()+self.mean_pixel, 0, 255))
          
      sess.run(self.synth_arr.assign(tf.clip_by_value(
               self.synth_arr + self.mean_pixel, 0, 255)-self.mean_pixel))
      
      print("time:", "{:4.1f}".format(time.time()-start),
            "loss post-clip:","{:,}".format(sess.run(self.loss)))    
      
          
      self.img_out = (self.synth_arr[0].eval()+self.mean_pixel).astype('uint8')
      
      plt.figure(figsize=(20,40))
      plt.imshow(self.img_out)

      imsave((self.path + savename + '.png'), self.img_out)