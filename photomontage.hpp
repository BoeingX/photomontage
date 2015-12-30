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
