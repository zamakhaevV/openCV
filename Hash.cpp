#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img;
	Size size;

	int hyperVec[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };


	img = imread("C:\\Users\\Vlad\\Desktop\\forest.jpg");


	if (img.empty()){
		cout << "failed to open file" << endl;
	}
	else{
		cout << "file loaded OK" << endl;

		for (int y = 0; y < img.rows; ++y){		//height
			for (int x = 0; x < img.cols; ++x){	//width
				Scalar intensity = img.at<uchar>(y, x);
				cout << intensity << "\t";
			}
			cout << endl;
		}

		for (int y = 1; y < img.rows - 1; ++y){		//height
			for (int x = 1; x < img.cols - 1; ++x){	//width
				Scalar intensity = img.at<uchar>(y, x);
				Scalar temp[8] {
					img.at<uchar>(y - 1, x - 1), img.at<uchar>(y - 1, x), img.at<uchar>(y - 1, x + 1),
					img.at<uchar>(y    , x - 1),                          img.at<uchar>(y    , x + 1),
					img.at<uchar>(y + 1, x - 1), img.at<uchar>(y + 1, x), img.at<uchar>(y + 1, x + 1)
				};

				for (int k = 0; k < 8; ++k)
					hyperVec[k] += (intensity.val[0] <= temp[k].val[0] ? 1 : 0);
			}
		}
	}


	cout << "[ ";
	for (int i = 0; i < 7; ++i){
		cout << hyperVec[i];
		cout << " , ";
	}
	cout << hyperVec[7] <<" ]";
	cout << endl;


	system("pause");
	return 0;
}
