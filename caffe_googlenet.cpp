/**M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include <string>
using namespace std;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

static std::vector<String> readClassNames(const char *filename = "synset_words.txt")
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open()){
		std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof()){
        std::getline(fp, name);
        if (name.length())
			classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}


bool dog_groundtruth_checker(string imagename) // Check dog test class file and returns if instance is a dog.
	{
		string classpath = "dog_test.txt"; // ground truth file for dog class
		ifstream file(classpath);
		std::string line;
		if(!file.is_open()){
			cout << "file is not open"<<endl;
		}
		else{ //cout<< " file is open" <<endl;
			int c = 0;
			while(std::getline(file,line)){	 // loop through all the lines in the ground truth file 
				std::stringstream ss(line);
				string a, b;
				ss >> a >> b;		
				a = a+".jpg";
				if (!a.compare(imagename)){ // find the line of the same image name
					if(!b.compare("0") || !b.compare("1")){ 
						c++;
						return true;	 //It's a dog!!;
					}
					else if(!b.compare("-1")){
						c++;
						return false; //It's not a dog!!
					}
				}
				
			}
			if(c==0){ 
				std::cerr << "Picture not found in the ground truth file. " << imagename << endl;					
				exit(-1);				
			}
		}
	}


int main(int argc, char **argv)
{
    
	
	CV_TRACE_FUNCTION();

    String modelTxt = "bvlc_googlenet.prototxt";
    String modelBin = "bvlc_googlenet.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "space_shuttle.jpg";
    const char* directory = "/home/nvidia/Documents/VOCdevkit/VOC2007/tryout/"; // Directory with the images that you want to analyze
	
	std::vector<std::string> dogs = {"Chihuahua","Japanese spaniel","Maltese dog, Maltese terrier, Maltese","Pekinese, Pekingese, Peke","Shih-Tzu","Blenheim spaniel","papillon","toy terrier","Rhodesian ridgeback","Afghan hound, Afghan","basset, basset hound","beagle","bloodhound, sleuthhound","bluetick","black-and-tan coonhound","Walker hound, Walker foxhound","English foxhound","redbone","borzoi, Russian wolfhound","Irish wolfhound","Italian greyhound","whippet","Ibizan hound, Ibizan Podenco","Norwegian elkhound, elkhound","otterhound, otter hound","Saluki, gazelle hound","Scottish deerhound, deerhound","Weimaraner","Staffordshire bullterrier, Staffordshire bull terrier","American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier","Bedlington terrier","Border terrier","Kerry blue terrier","Irish terrier","Norfolk terrier","Norwich terrier","Yorkshire terrier","wire-haired fox terrier","Lakeland terrier","Sealyham terrier, Sealyham","Airedale, Airedale terrier","cairn, cairn terrier","Australian terrier","Dandie Dinmont, Dandie Dinmont terrier","Boston bull, Boston terrier","miniature schnauzer","giant schnauzer","standard schnauzer","Scotch terrier, Scottish terrier, Scottie","Tibetan terrier, chrysanthemum dog","silky terrier, Sydney silky","soft-coated wheaten terrier","West Highland white terrier","Lhasa, Lhasa apso","flat-coated retriever","curly-coated retriever","golden retriever","Labrador retriever","Chesapeake Bay retriever","German short-haired pointer","vizsla, Hungarian pointer","English setter","Irish setter, red setter","Gordon setter","Brittany spaniel","clumber, clumber spaniel","English springer, English springer spaniel","Welsh springer spaniel","cocker spaniel, English cocker spaniel, cocker","Sussex spaniel","Irish water spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old English sheepdog, bobtail","Shetland sheepdog, Shetland sheep dog, Shetland","collie","Border collie","Bouvier des Flandres, Bouviers des Flandres","Rottweiler","German shepherd, German shepherd dog, German police dog, alsatian","Doberman, Doberman pinscher","miniature pinscher","Greater Swiss Mountain dog","Bernese mountain dog","Appenzeller","EntleBucher","boxer","bull mastiff","Tibetan mastiff","French bulldog","Great Dane","Saint Bernard, St Bernard","Eskimo dog, husky","malamute, malemute, Alaskan malamute","Siberian husky","dalmatian, coach dog, carriage dog","affenpinscher, monkey pinscher, monkey dog","basenji","pug, pug-dog","Leonberg","Newfoundland, Newfoundland dog","Great Pyrenees","Samoyed, Samoyede","Pomeranian","chow, chow chow","keeshond","Brabancon griffon","Pembroke, Pembroke Welsh corgi","Cardigan, Cardigan Welsh corgi","toy poodle","miniature poodle","standard poodle","Mexican hairless","timber wolf, grey wolf, gray wolf, Canis lupus","white wolf, Arctic wolf, Canis lupus tundrarum","red wolf, maned wolf, Canis rufus, Canis niger","coyote, prairie wolf, brush wolf, Canis latrans","dingo, warrigal, warragal, Canis dingo","dhole, Cuon alpinus","African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus"}; 
	// Array with all the dog species.

    //! [Read and initialize network]
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    //! [Read and initialize network]

    //! [Check that network was read successfully]
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
        std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
        exit(-1);
    }
    //! [Check that network was read successfully]
	
	double true_positives = 0.0; // Initialize the counters needed for the accuracy formula.
	double false_positives = 0.0;
	double true_negatives = 0.0;
	double false_negatives = 0.0;

	

	std::vector<std::string> pictureNames;	// vector with the names of the pictures in the given directory
	std::map<std::string,std::string> dictionary;  // Map that contains the picture name and it's predicted class.
	DIR *dirFile = opendir(directory);
	if (dirFile) 
	{
		struct dirent *hFile;
		errno = 0;
		while (( hFile = readdir(dirFile)) != NULL ){ // Loop through the pictures in the given directory and store their names.
			if ( strstr( hFile->d_name, ".jpg" )){
				string pictureName;
				pictureName = hFile->d_name;
				pictureNames.push_back(pictureName);
			}
		} 
		closedir(dirFile);
	}

    //! [Prepare blob]
	for (int i = pictureNames.size()-1; i>=0;i--){		// start prediction loop for all pictures in the directory
		imageFile = directory + pictureNames[i];
		Mat img = imread(imageFile);
		if (img.empty()){
			std::cerr << "Can't read image from the file: " << imageFile << std::endl;
    	    exit(-1);
    	}
		
		//GoogLeNet accepts only 224x224 RGB-images
    	Mat inputBlob = blobFromImage(img, 1, Size(224, 224), Scalar(104, 117, 123));   //Convert Mat to batch of images
    	//! [Prepare blob]
		Mat prob;
    	cv::TickMeter t;
    	for (int i = 0; i < 10; i++)
    	{
        	CV_TRACE_REGION("forward");
        	//! [Set input blob]
        	net.setInput(inputBlob, "data");        //set the network input
        	//! [Set input blob]
        	t.start();
        	//! [Make forward pass]
        	prob = net.forward("prob");       //compute output
        	//! [Make forward pass]
        	t.stop();
    	}
		//! [Gather output]
    	int classId;
    	double classProb;
    	getMaxClass(prob, &classId, &classProb);//find the best class
    	//! [Gather output]
		std::vector<String> classNames = readClassNames();
	
		// if class is a dog specy, replace the class with "dog"		
		for(int z = dogs.size()-1; z>=0; z--){		
			if(classNames.at(classId) == dogs[z]){
				dictionary[pictureNames[i]] = "dog";
				break;
			}
			else{			
				dictionary[pictureNames[i]] = classNames.at(classId); 
			}
		}   	

	// This block tracks if the prediction was correct or wrong.
	if (dictionary[pictureNames[i]] == "dog"){ // predicted to be a dog
		if (dog_groundtruth_checker(pictureNames[i]) == true){ // it actualy is a dog
			true_positives++;
		}else 
		if (dog_groundtruth_checker(pictureNames[i]) == false){ // it's not a dog
			false_positives++;
		}
	}else 
	if (dictionary[pictureNames[i]] != "dog"){ // predicted not a dog
		if (dog_groundtruth_checker(pictureNames[i]) == false){ // it's not a dog
			true_negatives++;
		}else 
		if(dog_groundtruth_checker(pictureNames[i]) == true){ // it actually is a dog
			false_negatives++;
		}
	}

	double sum = true_positives +false_positives +true_negatives +false_negatives;

	std::cerr << pictureNames[i] << " is classified as / " << dictionary[pictureNames[i]] << std::endl;
	std::cerr << pictureNames[i] << " ground truth dog? / " << dog_groundtruth_checker(pictureNames[i]) << std::endl;
	std::cerr  << "SUM = " << sum << " | TP = " << true_positives  << " | True Negatives  = " << true_negatives  << " | FP = " << false_positives << " | FN = " << false_negatives <<std::endl;

} // end of loop over all images
   
// calculation of the accuracy (TP+TN)/(TP+FP+TN+FN)*100% 
double trues = true_positives + true_negatives;
double all = true_positives + true_negatives + false_positives + false_negatives;
double accuracy = trues / all;
accuracy = accuracy * 100.0;

//FINAL SCORE
std::cerr << "True Positives  | " << true_positives <<std::endl;
std::cerr << "True Negatives  | " << true_negatives <<std::endl;
std::cerr << "False Positives | " << false_positives <<std::endl;
std::cerr << "False Negatives | " << false_negatives <<std::endl;
std::cerr << "Predictor's accuracy with the classification of dogs is: " << accuracy << "%" <<std::endl;

    return 0;
} //main
