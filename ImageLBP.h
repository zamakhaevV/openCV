#pragma once // �������� ���� ��� ���������� ����������� ������ ���� ���

#include "ImageDescriptor.h"
#include "PixelLBP.h"

class ImageLBP : public ImageDescriptor{
public:
	void getImageLBP(float _result[], PixelLBP pixel);
};