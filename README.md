# SinGAN
[Project](http://webee.technion.ac.il/people/tomermic/SinGAN/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) 
### Official implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image"
####  ICCV 2019

pytorch imlementation for SinGAN, which learns a generative model from a single image, for example:

![](imgs/teaser.PNG)


####  train
The train SinGAN model on your own training image, put the desire single training image under Input/Images, and run

'''
python train.py --input_name <input_file_name>
'''

This will also generate random samples starting from the coarser scale.

####  generate random samples


####  generate random samples of arbitrery sizes


####  generate animation from a single image

