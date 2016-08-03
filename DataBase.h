#pragma once // »сходный файл при компил€ции подключалс€ строго один раз


class DataBase{
private:
	const float GOOD_LABEL = 1;
	const float BAD_LABEL = -1;

	std::string GOOD_PATH;
	std::string BAD_PATH;

	float *labels;
	float *data;

	int samples_amount;

	int upload(std::string path, std::vector<std::string> *files);
public:
	DataBase(std::string good_path, std::string bad_path);
	~DataBase();

	float *getData();
	float *getLabels();
	int getSamplesAmount();
	float getGoodLabel();
	float getBadLabel();

	void createDB();
	void printDB();
};