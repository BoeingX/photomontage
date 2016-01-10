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

Mat read(char *filename);
void write(const Mat &img, char *filename);

void showNaive(const Mat &img1, const Mat &img2, const pair<int, int> offset);
void show(const Mat &img1, const Mat &img2, const Point2f &ul, const Point2f &lr);

pair<int, int> offset(const Mat &img1, const Mat &img2, set<Point2f> &hist, const int method);

void closure(const vector<Point2f> &pts, Point2f &ul, Point2f &lr);
Mat homography(const Mat &img1, const Mat &img2);
Mat calibration(const Mat &img1, const Mat &img2);
pair<int, int> homoMatching(const Mat &img1, const Mat &img2);

pair<int, int> entirePatchMatching(const Mat &img1, const Mat &img2, set<Point2f> &hist, const float minPortionH = 0.f,
                                   const float minPortionV = 0.f);
void entirePatchMatchingTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score, set<Point2f> &hist,
                            const float minPortionH = 0.f, const float minPortionV = 0.f);
