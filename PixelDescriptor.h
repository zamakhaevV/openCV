#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "opencv2\opencv.hpp"


class PixelDescriptor{
protected:
	cv::Mat image;
	int x, y;

	virtual void algorithm(float _binary[]) = 0; // ѕереопредел€етс€ в подклассах

public:
	void setXY(int _x, int _y);
	void setAll(cv::Mat _image, int _x, int _y);
	void setImage(cv::Mat _image);

	int getX();
	int getY();
	cv::Mat getMat();
};