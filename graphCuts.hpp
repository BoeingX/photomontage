#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "maxflow/graph.h"
using namespace std;
using namespace cv;

void gradient(const Mat& I, Mat& grad_x, Mat& grad_y, Mat& grad);
double matchingCost(const Point2i& s1, const Point2i& t1, const Point2i& s2, const Point2i& t2, const Mat& img1, const Mat& img2);

void correspondance(const Point2i& pt, Point2i& pt1, Point2i& pt2, const Point2i& offset, const Mat& img1, const Mat& img2);

void graphCut(Mat& output, const Mat& img1, const Mat& img2, const Point2i& offset);

Mat synthesis(const Mat& img1, const Mat& img2, const Point2i& offset);

