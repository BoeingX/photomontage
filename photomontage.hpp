#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#define SIGGRAPH 0
#define PANORAMA 1
using namespace std;
using namespace cv;

void display(const char *name, Mat img, int showTime = 0);
Mat showNaive(const Mat &img1, const Mat &img2, const Point2i offset);
Point2i offset(const Mat &img1, const Mat &img2, set<Point2f> &hist, const int method);
void closure(const vector<Point2f> &pts, Point2f &ul, Point2f &lr);
Mat homography(const Mat &img1, const Mat &img2);
Mat relugarization(const Mat &img1, const Mat &img2);
Point2i homoMatching(const Mat &img1, const Mat &img2);

Point2i entirePatchMatching(const Mat &img1, const Mat &img2, set<Point2f> &hist);
void entirePatchMatchingTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score, set<Point2f> &hist);
