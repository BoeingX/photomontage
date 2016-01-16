#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <limits>

#include "photomontage.hpp"
#include "maxflow/graph.h"

using namespace std;
using namespace cv;

// calculate intensity of gradients Ix and Iy of image I, containing all channels;
void gradient(const Mat& I, Mat& grad_x, Mat& grad_y){
	Mat tmp;
	Sobel(I, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(I, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
}

// compute matching cost according to formula (1);
double matchingCost(const Point2i& s1, const Point2i& t1, const Point2i& s2, const Point2i& t2, const Mat& img1, const Mat& img2){
	return norm(img1.at<Vec3b>(s1)-img2.at<Vec3b>(s2)) + norm(img1.at<Vec3b>(t1)-img2.at<Vec3b>(t2));
}

// return respectively points of img1 and img2, located in the overlap, corresponding to the point of the output;
void correspondance(const Point2i& pt, Point2i& pt1, Point2i& pt2, const Point2i& offset, const Mat& img1, const Mat& img2){
	int dx = offset.x;
	int dy = offset.y;
	if (dy >= 0){
		pt1.x = pt.x;
		pt1.y = pt.y;
		pt2.x = pt.x - dx;
		pt2.y = pt.y - dy;
	}
	else{
		pt1.x = pt.x;
		pt1.y = pt.y + dy;
		pt2.x = pt.x - dx;
		pt2.y = pt.y;
	}
}

// construct graph;
void graphCut(Mat& output, const Mat& img1, const Mat& img2, const Point2i& offset){
	if (offset.x >= img1.cols || offset.y >= img1.rows || offset.y <= -1 * img2.rows){
		cout << "Error: offset too great!" << endl;
		return;
	}
	// define infinity;
	double infinity = std::numeric_limits<double>::infinity();
	Point2i pt1, pt2;
	// compute overlap region according to two cases: offset.y>=0 and otherwise;
	if (offset.y >= 0){
		const Point2i upperLeft(offset);
		// define two extreme points of the overlap region;
		const Point2i lowerRight(min(img1.cols-1, offset.x+img2.cols-1), min(img1.rows-1, offset.y+img2.rows-1));
		int cols = lowerRight.x - upperLeft.x+1, rows = lowerRight.y - upperLeft.y+1;
		int numNodes = (lowerRight.x - upperLeft.x+1)*(lowerRight.y - upperLeft.y+1);
		Graph<double, double, double> g(numNodes, numNodes*6);
		// initialize nodes;
		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			for (int j = upperLeft.x; j <= lowerRight.x; j++){
				g.add_node(1);				
			}
		}
		
		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			for (int j = upperLeft.x; j <= lowerRight.x; j++){
				
				correspondance(Point2i(j, i), pt1, pt2, offset, img1, img2);
				
				// if adjacent to img1 or img2, set cost to be infinity;
				if (j == upperLeft.x){
					g.add_tweights(j-upperLeft.x + cols*(i-upperLeft.y), infinity, 0);
				}
				if (j == lowerRight.x){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), 0, infinity);
				}

				if (i==upperLeft.y && (j>upperLeft.x && j < lowerRight.x)){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), infinity, 0);
				}
				if (i == lowerRight.y && (j>upperLeft.x && j < lowerRight.x)){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), 0, infinity);
				}
				// set cost between nodes other than source or sink;
				double cost=0;
				if (j < lowerRight.x){
					cost = matchingCost(pt1, pt1 + Point2i(1, 0), pt2, pt2 + Point2i(1, 0), img1, img2);
					//cost = cost / (norm(grad_x1.at<double>(pt1)) + norm(grad_x1.at<double>(pt1 + Point2i(1, 0))) + norm(grad_x2.at<double>(pt2)) + norm(grad_x2.at<double>(pt2 + Point2i(1, 0))));
					g.add_edge(j-upperLeft.x + (i-upperLeft.y)*cols, j-upperLeft.x + 1 + (i-upperLeft.y)*cols, cost, cost);
				}
				if (i < lowerRight.y){
					cost = matchingCost(pt1, pt1 + Point2i(0, 1), pt2, pt2 + Point2i(0, 1), img1, img2);
					//cost = cost / (norm(grad_y1.at<double>(pt1)) + norm(grad_y1.at<double>(pt1 + Point2i(0, 1))) + norm(grad_y2.at<double>(pt2)) + norm(grad_y2.at<double>(pt2 + Point2i(0, 1))));
					g.add_edge(j-upperLeft.x + (i-upperLeft.y)*cols, j-upperLeft.x + (i-upperLeft.y + 1)*cols, cost, cost);
				}
				
			}
		}
		
		double flow = g.maxflow();
		cout << "Flow = " << flow << endl;
		for (int i = upperLeft.y; i < lowerRight.y; i++){
			for (int j = upperLeft.x; j < lowerRight.x; j++){
				if (g.what_segment(j-upperLeft.x + (i-upperLeft.y)*cols) == Graph<double, double, double>::SOURCE){
					Point2i pt1, pt2;
					correspondance(Point2i(j, i), pt1, pt2, offset, img1, img2);
					output.at<Vec3b>(i, j) = img1.at<Vec3b>(pt1);
				}
			}
		}
	}
	if (offset.y < 0){
		const Point2i upperLeft(offset.x, -1*offset.y);
		const Point2i lowerRight(min(img1.cols-1, offset.x+img2.cols-1), min(img2.rows-1, -1*offset.y+img1.rows-1));
		int cols = lowerRight.x - upperLeft.x + 1, rows = lowerRight.y - upperLeft.y + 1;
		int numNodes = (lowerRight.x - upperLeft.x+1)*(lowerRight.y - upperLeft.y+1);
		Graph<double, double, double> g(numNodes, numNodes * 6);
		// initialize nodes;
		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			for (int j = upperLeft.x; j <= lowerRight.x; j++){
				g.add_node(1);
			}
		}

		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			for (int j = upperLeft.x; j <= lowerRight.x; j++){

				correspondance(Point2i(j, i), pt1, pt2, offset, img1, img2);
				
				// if adjacent to img1 or img2, set cost to be infinity;
				if (j == upperLeft.x){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), infinity, 0);
				}
				
				if (j == lowerRight.x){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), 0, infinity);
				}
				
				if (i == upperLeft.y && (j>upperLeft.x && j < lowerRight.x)){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), 0, infinity);
				}
				
				if (i == lowerRight.y && (j>upperLeft.x && j < lowerRight.x)){
					g.add_tweights(j - upperLeft.x + cols*(i - upperLeft.y), infinity, 0);
				}
				
				// set cost between nodes other than source or sink;
				double cost = 0;
				if (j < lowerRight.x){
					cost = matchingCost(pt1, pt1 + Point2i(1, 0), pt2, pt2 + Point2i(1, 0), img1, img2);
					g.add_edge(j - upperLeft.x + (i - upperLeft.y)*cols, j - upperLeft.x + 1 + (i - upperLeft.y)*cols, cost, cost);
				}
				if (i < lowerRight.y){
					//cost = matchingCost(pt1, pt1 + Point2i(0, 1), pt2, pt2 + Point2i(0, 1), img1, img2);
					g.add_edge(j - upperLeft.x + (i - upperLeft.y)*cols, j - upperLeft.x + (i - upperLeft.y + 1)*cols, cost, cost);
				}

			}
		}

		double flow = g.maxflow();
		cout << "Flow = " << flow << endl;
		for (int i = upperLeft.y; i < lowerRight.y; i++){
			for (int j = upperLeft.x; j < lowerRight.x; j++){
				if (g.what_segment(j - upperLeft.x + (i - upperLeft.y)*cols) == Graph<double, double, double>::SOURCE){
					Point2i pt1, pt2;
					correspondance(Point2i(j, i), pt1, pt2, offset, img1, img2);
					output.at<Vec3b>(i, j) = img1.at<Vec3b>(pt1);
				}
			}
		}
		
	}
	
}

// synthesize images using graph cuts;
Mat showGraphCut(const Mat &img1, const Mat &img2, const Point2i &offset){
	Mat output = showNaive(img1, img2, offset);
    if(offset.x < 0)
	    graphCut(output, img2, img1, Point2i(-offset.x, -offset.y));
    else
	    graphCut(output, img1, img2, offset);
	return output;
}

/*
int main() {
	//testGCuts();
	Mat img1, img2, img3, img4, img5, output;
	img1=imread("../image0006.jpg"); //CV_8UC3;
	img2 = imread("../image0007.jpg");
	img3 = imread("../image0008.jpg");
	img4 = imread("../IMG_0026.JPG");
	img5 = imread("../IMG_0027.JPG");

	imshow("I1", img1);
	imshow("I2", img2);
	imshow("I3", img3);

	Point2i offset1(454, -8);
	output = synthesis(img1, img2, offset1);
	Point2i offset2(6, -314);
	output = synthesis(output, img3, offset2);

	Point2i offset(366, -9);
	output = showGraphCut(img4, img5, offset);
	imshow("Output", output);
	waitKey();
	return 0;
}
*/
