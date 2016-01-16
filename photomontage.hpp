#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#define SIGGRAPH 0
#define PANORAMA 1

using namespace std;
using namespace cv;

// display an image with an appropriate window
void display(const char *name, Mat img, int showTime = 0);

// find the closure of a rotated rectangle defined by four points
void closure(const vector<Point2f> &pts, Point2f &ul, Point2f &lr);

// find the inscribe rectangle of a rotated rectangle 
// defined by four points
void inscribe(const vector<Point2f> &pts, Point2f &ul, Point2f &lr);

// return a composed image of img1 and img2 
// given an offset with offset.x >= 0
Mat showNaivePositive(const Mat &img1, const Mat &img2, const Point2i offset);

// return a composed image of img1 and img2 of an given offset
Mat showNaive(const Mat &img1, const Mat &img2, const Point2i offset);

// return the homography matrix between img1 and img2
// using AKAZE method
Mat homography(const Mat &img1, const Mat &img2);

// return offset of img2 with respected to img1
// since img2 is deformed by (reverse) homography matrix
// we do a post processing so that
// the transformed img2 is centered and no black boundary around
Point2i homoMatching(const Mat &img1, const Mat &img2, Mat &img2Regu);

// entire patch matching algorithm
Point2i entirePatchMatching(const Mat &img1, const Mat &img2);

// entire matching algorithm
// supposing that img2 is at the right down direction of img1
void entirePatchMatchingTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score);
