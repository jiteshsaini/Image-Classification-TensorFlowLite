<p align="left">
<a href='https://helloworld.co.in/article/image-classification-tensorflow-lite-explained' target='_blank'>
   Read the Article
</a> 

Youtube Channel: <a href='https://www.youtube.com/channel/UC_2OyRNVCWCH8ipgmAoJ1mA' target='_blank'>
   <img src='https://github.com/jiteshsaini/files/blob/main/img/btn_youtube_2.png' height='20px'>
</a>
</p>

# Image Classification using TensorFlowLite on Raspberry Pi

A simple python script to demonstrate working of Image Classification on Raspberry Pi

<img src='https://github.com/jiteshsaini/files/blob/main/img/image-classification-tensorflow-lite-raspberry-pi.jpeg'>

## sample_pictures
This folder contains pictures of different objects/ animals. You can add more photographs of objects around you and test the script.

## model_files
This folder contains 03 different Pre-trained Models. All these models perform Image Classification and use a common label file. The label file conatain a list of 1000 objects. 
The Models are trained to classify these 1000 objects. The script uses one Model at a time to perform inference. You can change the Model in the script.

## requirements.sh
Install the pre-requisites i.e. Tensorflow Lite on Raspberry Pi using this bash script. Use the command 'sudo sh requirements.sh' in Terminal to install.

## classify.py
The Python script with Image Classification code.
- Uncomment line 12 or 13 to use a different Model
- Use a different image by changing image name at line 44.
