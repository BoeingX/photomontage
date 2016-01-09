#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
Mat read(char *filename);
void write(const Mat &img, char *filename);
void closure(const Mat &img, const Mat &H, int &minX, int &maxX, int &minY, int &maxY);
Mat homography(const Mat &img1, const Mat &img2);
Mat glue(const Mat &img1, const Mat &img2);
pair<int, int> bestOffset(const Mat &img1, const Mat &img2, const float minPortionH = 0.f, const float minPortionV = 0.f);
void bestOffsetTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score, const float minPortionH = 0.f, const float minPortionV = 0.f);
