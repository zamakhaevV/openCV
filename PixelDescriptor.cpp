#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "PixelDescriptor.h"


void PixelDescriptor::setXY(int _x, int _y){
	x = _x;
	y = _y;
}

void PixelDescriptor::setImage(cv::Mat _image){
	image = _image;
}

void PixelDescriptor::setAll(cv::Mat _image, int _x, int _y){
	image = _image;
	x = _x;
	y = _y;
}

int PixelDescriptor::getX(){
	return x;
}
int PixelDescriptor::getY(){
	return y;
}
cv::Mat PixelDescriptor::getMat(){
	return image;
}