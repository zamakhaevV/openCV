#pragma once // �������� ���� ��� ���������� ����������� ������ ���� ���

#include "opencv2\opencv.hpp"


class PixelDescriptor{
protected:
	cv::Mat image;
	int x, y;

	virtual void algorithm(float _binary[]) = 0; // ���������������� � ����������

public:
	void setXY(int _x, int _y);
	void setAll(cv::Mat _image, int _x, int _y);
	void setImage(cv::Mat _image);

	int getX();
	int getY();
	cv::Mat getMat();
};