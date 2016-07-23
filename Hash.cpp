#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int DIMENSION	= 8;
const int BAD_AMOUNT	= 150;   //Количество плохих срабатываний
const int GOOD_AMOUNT	= 173;   //Количество хороших срабатываний

const string MAIN_PATH		= "C:\\Users\\Vlad\\Desktop\\Classes\\";
const string GOOD_PATH		= "Good\\";
const string BAD_PATH		= "Bad\\";
const string PHOTO_FORMAT	= ".jpg";


void LBP_for_pixel(int x, int y, int a[], Mat img);
void LBP_for_image(int a[], Mat img);
void LBP_print(int a[]);
void create_database(int label[], int data[][DIMENSION]);
void print_database(int label[], int data[][DIMENSION]);



int main()
{
	int label[BAD_AMOUNT + GOOD_AMOUNT];
	int data[BAD_AMOUNT + GOOD_AMOUNT][DIMENSION];

	create_database(label, data);
	print_database(label, data);

	system("pause");
	return 0;
}


void LBP_for_pixel(int x, int y, int a[], Mat img)
{
	int tmp[DIMENSION] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	Scalar intensity = img.at<uchar>(y, x);
	Scalar temp[DIMENSION] {
		img.at<uchar>(y - 1, x - 1), img.at<uchar>(y - 1, x), img.at<uchar>(y - 1, x + 1),
		img.at<uchar>(y    , x - 1),                          img.at<uchar>(y    , x + 1),
		img.at<uchar>(y + 1, x - 1), img.at<uchar>(y + 1, x), img.at<uchar>(y + 1, x + 1)
	};

	for (int k = 0; k < DIMENSION; ++k){
		tmp[k] += (intensity.val[0] <= temp[k].val[0] ? 1 : 0);
		a[k] = tmp[k];
	}
}

void LBP_for_image(int a[], Mat img)
{
	int result[DIMENSION] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int y = 1; y < img.rows - 1; ++y){		//height
		for (int x = 1; x < img.cols - 1; ++x){	//width
			LBP_for_pixel(x, y, a, img);
			for (int k = 0; k < DIMENSION; ++k)
				result[k] += a[k];
		}
	}

	for (int k = 0; k < DIMENSION; ++k){
		a[k] = result[k];
	}

}

void LBP_print(int a[]){
	cout << "[ ";
	for (int i = 0; i < DIMENSION - 1; ++i){
		cout << a[i];
		cout << " , ";
	}
	cout << a[DIMENSION - 1] << " ]";
	cout << endl;
}

void create_database(int label[], int data[][DIMENSION]){
	int hyperVec[DIMENSION];
	Mat img;

	for (int i = 0; i < BAD_AMOUNT + GOOD_AMOUNT; ++i){
		label[i] = 0;

		for (int j = 0; j < DIMENSION; ++j){
			data[i][j] = 0;
		}
	}

	for (int i = 0; i < GOOD_AMOUNT; ++i){ //Нумерация фотографий с нуля
		label[i] = 1;
		img	 = imread(MAIN_PATH + GOOD_PATH + to_string(i) + PHOTO_FORMAT);

		LBP_for_image(hyperVec, img);
		LBP_print(hyperVec);

		for (int k = 0; k < DIMENSION; ++k)
			data[i][k] = hyperVec[k];

	}


	for (int i = GOOD_AMOUNT, j = 0; i < GOOD_AMOUNT + BAD_AMOUNT; ++i, ++j){ //Нумерация фотографий с нуля
		label[i] = -1;
		img	 = imread(MAIN_PATH + BAD_PATH + to_string(j) + PHOTO_FORMAT);

		LBP_for_image(hyperVec, img);
		LBP_print(hyperVec);

		for (int k = 0; k < DIMENSION; ++k)
			data[i][k] = hyperVec[k];

	}
}

void print_database(int label[], int data[][DIMENSION]){
	for (int i = 0; i < BAD_AMOUNT + GOOD_AMOUNT; ++i){
		cout << "[ ";
		cout << label[i];
		cout << "\t | ";

		for (int j = 0; j < DIMENSION; ++j){
			cout << " , ";
			cout << data[i][j];
		}

		cout << " ]";
		cout << endl;
	}
}
