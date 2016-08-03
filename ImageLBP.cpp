#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "ImageLBP.h"

void ImageLBP::getImageLBP(float _result[], PixelLBP pixel){
	float LBP[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	for (int i = 0; i < 8; ++i)
		_result[i] = 0;

	cv::Mat image = pixel.getMat();

	for (int y = 1; y < image.rows - 1; ++y){		//height
		for (int x = 1; x < image.cols - 1; ++x){	//width
			pixel.getLBP(LBP, x, y);
			for (int i = 0; i < 8; ++i)
				_result[i] += LBP[i];
		}
	}
}