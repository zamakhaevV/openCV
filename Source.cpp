#include <iostream>
#include <string>
#include "dirent.h"
#include "opencv2\opencv.hpp"

#include "DataBase.h"
#include "PixelLBP.h"
#include "ImageLBP.h"

using std::cout;
using std::endl;


std::string GOOD_PATH = "C:\\Users\\Vlad\\Desktop\\Classes\\Good\\";
std::string BAD_PATH = "C:\\Users\\Vlad\\Desktop\\Classes\\Bad\\";
std::string HAAR_PATH = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_fist.xml";


void train(float *labels, float *data, int samples_count, int data_dim, CvSVM *dst);
float prediction(const CvSVM *svm, cv::Mat img);

void help();

cv::Rect template_func(cv::Mat img, cv::Mat templ, int match_method);
cv::RotatedRect camshift_func(cv::Mat img, cv::Rect selection);
void lk_func(cv::Mat img, bool *needToInit);

int main()
{
	help();

	bool fix = false;

	DataBase DB(GOOD_PATH, BAD_PATH);
	DB.createDB();

	CvSVM svm;
	train(DB.getLabels(), DB.getData(), DB.getSamplesAmount(), 8, &svm);

	cv::VideoCapture capture(0);
	cv::Mat frame;
	cv::CascadeClassifier haar(HAAR_PATH);

	cv::Rect template_rect;
	cv::Mat temp;

	while (true)
	{
		capture >> frame;


		std::vector<cv::Rect>detects;
		haar.detectMultiScale(frame, detects, 1.2, 5, 0, cv::Size(40, 40), cv::Size(150, 150));


		for (int i = 0; i < detects.size(); i++){
			if (prediction(&svm, frame(detects[i])) == DB.getGoodLabel()){
				if (fix){
					template_rect = detects[i];
					temp = frame(detects[i]);
					fix = false;
				}

				cv::ellipse(frame, camshift_func(frame, template_rect), cv::Scalar(0, 0, 255), 3, CV_AA);
				cv::rectangle(frame, detects[i], cv::Scalar(255, 255, 0), 3);
			}
		}


		cv::imshow("Test", frame);
		char c = (char)cv::waitKey(10);
		if (c == 27)
			break;
		switch (c){
		case 'f':
			fix = true;
			break;
		default:
			;
		}
	}


	system("pause");
	return 0;
}

void train(float *labels, float *data, int samples_count, int data_dim, CvSVM *dst) {

	cv::Mat labels_mat(samples_count, 1, CV_32FC1, labels);
	cv::Mat data_mat(samples_count, data_dim, CV_32FC1, data);

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	dst->train(data_mat, labels_mat, cv::Mat(), cv::Mat(), params);
}

float prediction(const CvSVM *svm, cv::Mat img){
	float tmp[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	PixelLBP pix(img);
	ImageLBP image;

	image.getImageLBP(tmp, pix);

	cv::Mat sample_mat(8, 1, CV_32FC1, tmp);

	return svm->predict(sample_mat);
}

void help(){
	cout << "Hot keys: " << endl;
	cout << "\tESC - quit the program" << endl;
	cout << "\tf - fix fist" << endl;
}

cv::Rect template_func(cv::Mat img, cv::Mat templ, int match_method){
	cv::Mat result;

	/// Create the result matrix
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	/// Localizing the best match with minMaxLoc
	double minVal, maxVal;
	cv::Point minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED){
		matchLoc = minLoc;
	}
	else{
		matchLoc = maxLoc;
	}

	cv::Rect template_rect(matchLoc*0.9, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows)*1.1);

	return template_rect;
}

cv::RotatedRect camshift_func(cv::Mat img, cv::Rect selection){
	int trackObject = 0;
	int vmin = 128, vmax = 256, smin = 18;

	cv::Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;

	cv::Mat hsv, hue, mask, hist, histimg = cv::Mat::zeros(200, 320, CV_8UC3), backproj;

	if (selection.width > 0 && selection.height > 0)
		trackObject = -1;

	cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	if (trackObject)
	{
		int _vmin = vmin, _vmax = vmax;

		inRange(hsv, cv::Scalar(0, smin, MIN(_vmin, _vmax)), cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		int ch[] = { 0, 0 };
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);

		if (trackObject < 0)
		{
			cv::Mat roi(hue, selection), maskroi(mask, selection);
			calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
			normalize(hist, hist, 0, 255, CV_MINMAX);

			trackWindow = selection;
			trackObject = 1;

			histimg = cv::Scalar::all(0);
			int binW = histimg.cols / hsize;
			cv::Mat buf(1, hsize, CV_8UC3);
			for (int i = 0; i < hsize; i++)
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize), 255, 255);
			cvtColor(buf, buf, CV_HSV2BGR);

			for (int i = 0; i < hsize; i++)
			{
				int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
				cv::rectangle(histimg, cv::Point(i*binW, histimg.rows), cv::Point((i + 1)*binW, histimg.rows - val), cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
			}
		}

		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;
		cv::RotatedRect trackBox = CamShift(backproj, trackWindow, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

		if (trackWindow.area() <= 1)
		{
			int cols = backproj.cols;
			int rows = backproj.rows;
			int r = (MIN(cols, rows) + 5) / 6;

			trackWindow = cv::Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & cv::Rect(0, 0, cols, rows);
		}

		return trackBox;
	}
}