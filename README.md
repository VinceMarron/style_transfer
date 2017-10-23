# Style Transfer as Optimal Transport 

## An algorithm that transfers the distribution of visual characteristics, or *style*, of a reference image onto a subject image via an Optimal Transport plan. Implemented in TensorFlow. 


>"What features and statistics are characteristics of a texture pattern, so that texture pairs that share the same features and statistics cannot be told apart by pre-attentive human visual perception?” -- [Bela Julesz](https://en.wikipedia.org/wiki/B%C3%A9la_Julesz)


## 

Requires file with vgg19 weights (115.5 MB) 
can be downloaded [here] (https://app.box.com/v/vgg19-conv-npy)
                  md5sum: bf8a930fec201a0a2ade13d3f7274d0e"

`python basic_styletrans.py --subject media/small_surfer.jpg --style media/small_kanagawa.jpg --output media/small_surf_kngwa.jpg`
