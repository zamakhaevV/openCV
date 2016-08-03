#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "PixelLBP.h"


PixelLBP::PixelLBP(){
	image = cv::Mat::zeros(3, 3, CV_8U);
	x = -1;
	y = -1;
}

PixelLBP::PixelLBP(cv::Mat _image) {
	image = _image;
}

PixelLBP::PixelLBP(cv::Mat _image, int _x, int _y){
	image = _image;
	x = _x;
	y = _y;
}

void PixelLBP::getLBP(float _binary[], int _x, int _y){
	this->setXY(_x, _y);

	for (int i = 0; i < 8; ++i)
		_binary[i] = 0;
	algorithm(_binary);
}

void PixelLBP::algorithm(float _binary[]){
	if ( (0 < x) && (x < image.cols - 1) && (0 < y) && (y < image.rows - 1) ){
		cv::Scalar intensity = image.at<uchar>(y, x);
		cv::Scalar temp[8] {
			image.at<uchar>(y - 1, x - 1), image.at<uchar>(y - 1, x), image.at<uchar>(y - 1, x + 1),
				image.at<uchar>(y, x - 1), image.at<uchar>(y, x + 1),
				image.at<uchar>(y + 1, x - 1), image.at<uchar>(y + 1, x), image.at<uchar>(y + 1, x + 1)
		};

		for (int i = 0; i < 8; ++i)
			_binary[i] = (intensity.val[0] <= temp[i].val[0] ? (float) 1.0 : 0);
	}
	else
		std::cout << "Error! Extreme pixel" << std::endl;
}