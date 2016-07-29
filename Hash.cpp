#include <iostream>
#include <string>
#include <dirent.h>
#include <opencv2\opencv.hpp>

using std::cout;
using std::endl;


const int DIMENSION = 8;
const int SAMPLES_COUNT = 885;

const float GOOD_LABEL = 1;
const float BAD_LABEL = -1;

const char *GOOD_PATH = "C:\\Users\\Vlad\\Desktop\\Classes\\Good\\";
const char *BAD_PATH = "C:\\Users\\Vlad\\Desktop\\Classes\\Bad\\";
const char *HAAR_PATH = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_fist.xml";
const char *TEMPLATE_PATH = "C:\\Users\\Vlad\\Desktop\\template.jpg";


cv::Mat templ = cv::imread(TEMPLATE_PATH);
cv::Mat result, frame;

int match_method;
int max_Trackbar = 5;


bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
cv::Point origin;
cv::Rect selection;
int vmin = 10, vmax = 256, smin = 30;


void LBP_for_pixel(int x, int y, float a[], cv::Mat img);
void LBP_for_image(float a[], cv::Mat img);
void LBP_print(float a[]);
void create_database(float label[], float data[][DIMENSION]);
void print_database(float label[], float data[][DIMENSION]);
int dowload_photos(const char *path, std::vector<std::string> *files);

void train(float label[], float data[][DIMENSION], int samples_count, int data_dim, CvSVM* dst);
float prediction(const CvSVM* svm, cv::Mat img);

static void onMouse(int event, int x, int y, int, void*);
static void help();


int main()
{
	help();

	float label[SAMPLES_COUNT];
	float data[SAMPLES_COUNT][DIMENSION];

	create_database(label, data);

	CvSVM svm;
	train(label, data, SAMPLES_COUNT, DIMENSION, &svm);

	cv::VideoCapture capture(0);

	cv::CascadeClassifier haar(HAAR_PATH);

	cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);

	cv::Mat hsv, hue, mask, hist, histimg = cv::Mat::zeros(200, 320, CV_8UC3), backproj;


	cv::setMouseCallback("Test", onMouse, 0);
	cv::createTrackbar("Vmin", "Test", &vmin, 256, 0);
	cv::createTrackbar("Vmax", "Test", &vmax, 256, 0);
	cv::createTrackbar("Smin", "Test", &smin, 256, 0);

	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	int hsize = 16;
	cv::Rect trackWindow;

	while (true)
	{
		capture >> frame;

		/// Create the result matrix
		int result_cols = frame.cols - templ.cols + 1;
		int result_rows = frame.rows - templ.rows + 1;

		result.create(result_rows, result_cols, CV_32FC1);

		/// Do the Matching and Normalize
		matchTemplate(frame, templ, result, CV_TM_SQDIFF);
		normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
		cv::Point matchLoc;

		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
		if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
		{
			matchLoc = minLoc;
		}
		else
		{
			matchLoc = maxLoc;
		}


		std::vector<cv::Rect>detects;
		haar.detectMultiScale(frame, detects, 1.2, 5, 0, cv::Size(40, 40), cv::Size(150, 150));


		cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

		if (trackObject){
			int _vmin = vmin, _vmax = vmax;

			inRange(hsv, cv::Scalar(0, smin, MIN(_vmin, _vmax)), cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
			int ch[] = { 0, 0 };
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);

			if (trackObject < 0){
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

				for (int i = 0; i < hsize; i++){
					int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
					rectangle(histimg, cv::Point(i*binW, histimg.rows), cv::Point((i + 1)*binW, histimg.rows - val), cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
				}
			}

			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
			backproj &= mask;
			cv::RotatedRect trackBox = CamShift(backproj, trackWindow, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
			if (trackWindow.area() <= 1){
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = cv::Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & cv::Rect(0, 0, cols, rows);
			}

			if (backprojMode)
				cvtColor(backproj, frame, cv::COLOR_GRAY2BGR);
			cv::rectangle(frame, trackWindow, cv::Scalar(0, 0, 255), 3);

			for (int i = 0; i < detects.size(); i++){
				if (prediction(&svm, frame(detects[i])) == GOOD_LABEL)
					cv::rectangle(frame, detects[i], cv::Scalar(255, 255, 0), 3);

			}

			/// Show me what you got
			cv::Rect mL(matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows));
			cv::rectangle(frame, mL, cv::Scalar::all(0), 2, 8, 0);
		}

		if (selectObject && selection.width > 0 && selection.height > 0){
			cv::Mat roi(frame, selection);
			bitwise_not(roi, roi);
		}


		cv::imshow("Test", frame);
		char c = (char)cv::waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject = 0;
			histimg = cv::Scalar::all(0);
			break;
		default:
			;
		}

	}

	system("pause");
	return 0;
}

static void help()
{
	cout << "\n\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tc - stop the tracking\n"
		"\tb - switch to/from backprojection view\n"
		"To initialize tracking, select the object with mouse\n";
}

static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		selection &= cv::Rect(0, 0, frame.cols, frame.rows);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = cv::Point(x, y);
		selection = cv::Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;
		break;
	}
}

void LBP_for_pixel(int x, int y, float a[], cv::Mat img)
{
	float tmp[DIMENSION] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	cv::Scalar intensity = img.at<uchar>(y, x);
	cv::Scalar temp[DIMENSION] {
		img.at<uchar>(y - 1, x - 1), img.at<uchar>(y - 1, x), img.at<uchar>(y - 1, x + 1),
			img.at<uchar>(y, x - 1), img.at<uchar>(y, x + 1),
			img.at<uchar>(y + 1, x - 1), img.at<uchar>(y + 1, x), img.at<uchar>(y + 1, x + 1)
	};

	for (int k = 0; k < DIMENSION; ++k){
		tmp[k] += (intensity.val[0] <= temp[k].val[0] ? 1 : 0);
		a[k] = tmp[k];
	}
}

void LBP_for_image(float a[], cv::Mat img)
{
	float result[DIMENSION] = { 0, 0, 0, 0, 0, 0, 0, 0 };

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

void LBP_print(float a[]){
	cout << "[ ";
	for (int i = 0; i < DIMENSION - 1; ++i){
		cout << a[i];
		cout << " , ";
	}
	cout << a[DIMENSION - 1] << " ]";
	cout << endl;
}

void create_database(float label[], float data[][DIMENSION]){
	for (int i = 0; i < SAMPLES_COUNT; ++i){
		label[i] = 0;

		for (int j = 0; j < DIMENSION; ++j){
			data[i][j] = 0;
		}
	}

	cv::Mat img;
	std::vector<std::string> good_files;
	std::vector<std::string> bad_files;

	dowload_photos(GOOD_PATH, &good_files);
	for (int i = 0; i < good_files.size(); ++i){
		label[i] = GOOD_LABEL;
		img = cv::imread(GOOD_PATH + good_files[i]);

		LBP_for_image(data[i], img);
	}

	dowload_photos(BAD_PATH, &bad_files);
	for (int i = 0; i < bad_files.size(); ++i){
		label[i + good_files.size()] = BAD_LABEL;
		img = cv::imread(BAD_PATH + bad_files[i]);

		LBP_for_image(data[i + good_files.size()], img);
	}
}

void print_database(float label[], float data[][DIMENSION]){
	for (int i = 0; i < SAMPLES_COUNT; ++i){
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

int dowload_photos(const char *path, std::vector<std::string> *files){
	DIR *dir = opendir(path);
	struct dirent *ent;

	files->clear();

	readdir(dir); // ignore "."
	readdir(dir); // ignore ".."

	if (dir != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			files->push_back(ent->d_name);  // insert file's name into files

		}
		closedir(dir);
	}
	else {  // could not open directory
		perror("");
		return EXIT_FAILURE;
	}
	return 0;
}

void train(float label[], float data[][DIMENSION], int samples_count, int data_dim, CvSVM* dst) {
	cv::Mat label_mat(samples_count, 1, CV_32FC1, label);
	cv::Mat data_mat(samples_count, data_dim, CV_32FC1, data);

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	dst->train(data_mat, label_mat, cv::Mat(), cv::Mat(), params);
}

float prediction(const CvSVM* svm, cv::Mat img){
	float tmp[DIMENSION];

	LBP_for_image(tmp, img);
	cv::Mat sample_mat(DIMENSION, 1, CV_32FC1, tmp);

	return svm->predict(sample_mat);
}
