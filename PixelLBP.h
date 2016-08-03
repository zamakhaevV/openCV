#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "PixelDescriptor.h"


class PixelLBP : public PixelDescriptor{
protected:
	virtual void algorithm(float _binary[]) override;

public:
	PixelLBP();
	PixelLBP(cv::Mat _image);
	PixelLBP(cv::Mat _image, int _x, int _y);

	void getLBP(float _binary[], int _x, int _y);
};