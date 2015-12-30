#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
Mat read(string filename);
void write(const Mat &img, string filename);
void closure(Mat &img, const Mat &H);
void homography(Mat &img1, Mat &img2);
