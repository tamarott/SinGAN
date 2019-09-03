# SinGAN
[Project](http://webee.technion.ac.il/people/tomermic/SinGAN/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) 
### Official implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image"
####  ICCV 2019

pytorch imlementation for SinGAN, which learns a generative model from a single natural image. Using SinGAN you can create random samples form a given image, for example:

![](imgs/teaser.PNG)

<!--- 
// SinGAN can be also use to a line of image manipulation task, for example
 ![](imgs/manipulation.PNG)
 --->

###  train
To train SinGAN model on your own training image, put the desire single training image under Input/Images, and run

```
python train.py --input_name <input_file_name>
```


This will also generate random samples starting from the coarsest scale (n=0) of the resulting trained model.

###  generate random samples
To generate random samples from any starting generation scale, please first train SinGAN model for the desire image (as described above), then run 

```
python train.py --input_name <input_file_name> --mode random_samples --gen_start_scale <generation start scale number>
```

pay attention: for using the full model, specify the generation start scale number to be 0, to start the generation from the second scale, specify it to be 1, and so on. 

###  generate random samples of arbitrery sizes
To generate random samples of arbitrery sizes, please first train SinGAN model for the desire image (as described above), then run 

```
python train.py --input_name <input_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

###  generate animation from a single image

To generate random samples of arbitrery sizes, please first train SinGAN model for the desire image (as described above), then run 

```
python train.py --input_name <input_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```


### citation
If you use this code for your research, please cite our papers:

```
@inproceedings{shaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}
```

