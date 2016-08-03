#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include <iostream>
#include <string>
#include "opencv2\opencv.hpp"
#include "dirent.h"

#include "PixelLBP.h"
#include "ImageLBP.h"
#include "DataBase.h"


DataBase::DataBase(std::string good_path, std::string bad_path){
	labels = NULL;
	data = NULL;
	samples_amount = 0;
	GOOD_PATH = good_path;
	BAD_PATH = bad_path;
}

DataBase::~DataBase(){
	delete[] labels;
	delete[] data;

	labels = NULL;
	data = NULL;
	samples_amount = 0;
}

float DataBase::getGoodLabel(){
	return GOOD_LABEL;
}

float DataBase::getBadLabel(){
	return BAD_LABEL;
}

float *DataBase::getData(){
	return data;
}

float *DataBase::getLabels(){
	return labels;
}

int DataBase::getSamplesAmount(){
	return samples_amount;
}

int DataBase::upload(std::string path, std::vector<std::string> *files){
	DIR *dir = opendir(path.c_str());
	struct dirent *ent;

	files->clear();

	readdir(dir); // ignore "."
	readdir(dir); // ignore ".."

	if (dir != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			files->push_back(ent->d_name);  // insert file's name into files

		}
		closedir(dir);
	}
	else {  // could not open directory
		perror("");
		return EXIT_FAILURE;
	}
	return 0;
}

void DataBase::createDB(){
	PixelLBP pix;
	ImageLBP img;

	std::vector<std::string> good_files;
	std::vector<std::string> bad_files;

	DataBase::upload(GOOD_PATH, &good_files);
	DataBase::upload(BAD_PATH, &bad_files);

	samples_amount = good_files.size() + bad_files.size(); // размер массива

	float LBP[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	labels = new float[samples_amount];

	data = new float[samples_amount*8];


	for (int i = 0; i < good_files.size(); ++i){
		pix.setImage(cv::imread(GOOD_PATH + good_files[i]));
		img.getImageLBP(LBP, pix);

		labels[i] = GOOD_LABEL;

		for (int j = 0; j < 8; ++j)
			data[i*8 + j] = LBP[j];
	}

	for (int i = 0; i < bad_files.size(); ++i){
		pix.setImage(cv::imread(GOOD_PATH + good_files[i]));
		img.getImageLBP(LBP, pix);

		labels[i + good_files.size()] = BAD_LABEL;
		for (int j = 0; j < 8; ++j)
			data[(i + good_files.size())*8 + j] = LBP[j];
	}
}

void DataBase::printDB(){
	for (int i = 0; i < samples_amount; ++i){
		std::cout << "[ ";
		std::cout << labels[i];
		std::cout << "\t | ";

		for (int j = 0; j < 8; ++j){
			std::cout << " , ";
			std::cout << data[i*8 + j];
		}

		std::cout << " ]";
		std::cout << std::endl;
	}
}