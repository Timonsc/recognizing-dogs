# Recognizing dogs from ImageNet dataset.
Timon Schneider
Carlos Cajello
07 April 2018


With a Caffe GoogLeNet model running from a NVIDIA Jetson TX2 module we aimed to find all dogs in the ImageNet dataset. C++, OpenCV2.

#### Containing:
- unchanged caffemodel
 - unchanged prototxt
 - compiled caffe_googlenet file
 - caffe_googlenet source code 
 - ground truth file dog_test.txt 
 - set of images for testing purposes (does contain some tp, fp, tn and fn's)
 - synset_words.txt 

#### Notes:
 - All files except the tryout pictures were stored in /home/nvidia/samples/dnn/ when it was tested.
 - All images that are tested have to be in the ground truth file. Otherwise you get an error.

#### Compiled using the following line on the jetson: 
	gcc -std=c++11 caffe_googlenet.cpp -o caffe_googlenet -L/usr/lib/aarch64-linux-gnu -lstdc++ -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

#### How to run:
 - Set path to ground truth file (in this case dog_text.txt) on line 90
 - Set path to the image directory you want to text on line 132
 - put ./caffe_googlenet in commandline

#### What we have done:
This code aimes to check the accuracy of the caffe model on its prediction of the dog class. Since the model predicts for 1000 separate classes we chose to take all dog classes (more than 100) and count them as the mother class dog. Since a fair proportion of the images (and classes) contain dogs the accuracy will at least give a significant score for the prediction of the dog class. We wrote a code that loops the through the given directory of images. The prediction is for every image is checked by a method that checks the dog class ground truth file. Counters for TP, FP, TN and FN's are used to finally calculate the accuracy of the dog class prediction ( (TP + TN) / (TP + FP + TN + FN) * 100% ). The source code contains more specific comments.