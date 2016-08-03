#pragma once // »сходный файл при компил€ции подключалс€ строго один раз

#include "ImageDescriptor.h"
#include "PixelLBP.h"

class ImageLBP : public ImageDescriptor{
public:
	void getImageLBP(float _result[], PixelLBP pixel);
};