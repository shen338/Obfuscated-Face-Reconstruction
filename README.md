## Obfuscated Face Reconstruction
Estimating a high-resolution(HR) face image from its low-resolution(LR) counterpart using deep convolutional neural network. And using pretrained image classification model like VGGNet to contruct perceptual loss to obtain photo-realistic reconstruction. Plus, we may implement GAN(generative adversial network) and use LR images as prior info to reconstruct the fine details.  

## Team members
Tong Shen (shen338), Jieqiong Zhao (JieqiongZhao)

## Example of LR Masaiced Image(left) vs. HR Image(center) vs. LR Blurred Image(right)

 ![](https://raw.githubusercontent.com/shen338/DL/master/lowresimage-example.jpg)
 
## SRResNet and SRGAN models
#### SRResNet
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/SRResNet_model.PNG)
#### SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/GAN_model.PNG)

## SRResNet and SRGAN models
#### SRResNet 
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/SRResNet_result.PNG)
#### SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/SRGAN_result.PNG)
#### some failure case of SRGAN
![](https://raw.githubusercontent.com/shen338/Obfuscated-Face-Reconstruction/master/result/failure_case.PNG)

### contents
The original data path should be './origin/img_align_celeba_png/'. The image name is from 000001.png to 200000.png.
Run FaceCrop.py and TFtrain.py to generate TFRecord file. 

And train any model just run *_train.py file. All the records and checkpoints will be stored on tensorboard. 
