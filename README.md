## Obfuscated Face Reconstruction
Estimating a high-resolution(HR) face image from its low-resolution(LR) counterpart using deep convolutional neural network. And using pretrained image classification model like VGGNet to contruct perceptual loss to obtain photo-realistic reconstruction. Plus, we further implement GAN(generative adversial network) and use LR images as prior info to reconstruct the fine details.  

## Team members
Tong Shen (shen338), Jieqiong Zhao (JieqiongZhao)

## Example of LR Masaiced Image(left) vs. HR Image(center) vs. LR Blurred Image(right)

 ![](https://raw.githubusercontent.com/shen338/DL/master/lowresimage-example.jpg)
 
## SRResNet and SRGAN models
#### SRResNet
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/SRResNet_model.PNG)
#### SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/GAN_model.PNG)

## SRResNet and SRGAN results(random selected)
#### SRResNet 
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/SRResNet_result.PNG)
#### SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/GAN_result.PNG)
#### some failure case of SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/failure_case.PNG)

### contents
#1 Data preparation  
The original data path should be './origin/img_align_celeba_png/'. The image name is from 000001.png to 200599.png.
Run FaceCrop.py and TFtrain.py to generate TFRecord file.  
This will use CascadeClassifier from OpenCV to extract human faces and write a big binary file into disk for TensorFlow fast and parallelled reading. The effective data size is about 190000. 

#2 Use Deep nets to recontruct image from  its 1/4 size counterpart.  
And train any model just run *_train.py file. All the records and checkpoints will be stored on disk and can be viewed in TensorBoard. 

#3 Implement WGAN, WGANGP, BEGAN to recontruct image from  its 1/8 size counterpart.   
In this situation, the images are blurred too serious to be reconstructed. We use GAN(Generative Model) to generate information based on the whole dataset to compensate this. 
And train a GAN model run *_train.py file. All the records and checkpoints will be stored on disk and can be viewed in TensorBoard. And all the hyperparameters are listed at the beginning of code.  
