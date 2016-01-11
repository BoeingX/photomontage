#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <limits>

#include "maxflow/graph.h"

using namespace std;
using namespace cv;
float alpha = 1.0f, beta = 10000.0f;
char*  window_name = "Contour detection";

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      3    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 
	g.add_node(2); 
	g.add_tweights( 0,   /* capacities */  1, 5.1 );
	g.add_tweights( 1,   /* capacities */  6, 1.5 );
	g.add_edge( 0, 1,    /* capacities */  3, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}

// calculate intensity of gradients Ix, Iy and G of image I, containing all channels;
void forTest(const Mat& I, Mat& grad_x, Mat& grad_y){
	Mat tmp;
	Sobel(I, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(I, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, grad_x);	//Scales, calculates absolute values, and converts the result to 8-bit;
	convertScaleAbs(grad_y, grad_y);	//Scales, calculates absolute values, and converts the result to 8-bit;
}

void gradient(const Mat& I, Mat& grad_x, Mat& grad_y){
	Mat tmp;
	Sobel(I, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(I, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
}

/*
void cost(const Mat& I, Mat& gi, Mat& ge, const Vec3b& c1, const Vec3b& c2){
	Vec3f tmp1, tmp2;
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols; j++){
			tmp1 = c1;
			subtract(tmp1, I.at<Vec3b>(i, j), tmp1);
			gi.at<float>(i, j)= norm(tmp1);
			tmp2 = c2;
			subtract(tmp2, I.at<Vec3b>(i, j), tmp2);
			ge.at<float>(i, j) = norm(tmp2);

		}
	}
	convertScaleAbs(gi, gi);
	convertScaleAbs(ge, ge);
	return;
}

float GFun(const float x){
	return beta / (1 + alpha*x*x);
}

void onMouse1(int event, int x, int y, int foo, void* p){
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Mat* I = (Mat*)p;
	Point m1(x, y);
	cout << (*I).at<Vec3b>(m1) << endl;
	circle(*I, m1, 2, Scalar(0, 255, 0), 2);
	imshow("I", *I);
}

void GCuts(const Mat& I, const Mat& grad, const Mat& gi, const Mat& ge){
	Mat mask, I_fish, contour;
	mask = Mat::zeros(I.size(), CV_8UC1);
	contour = Mat::zeros(I.size(), CV_8UC1);
	Graph<float, float, float > g(I.cols*I.rows, I.cols*I.rows * 8);
	float cost = 0.0f;
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols; j++){
			g.add_node(1);
			g.add_tweights(j + I.cols*i, float(gi.at<uchar>(i, j)), float(ge.at<uchar>(i, j)));
		}
	}
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols; j++){
			if (j<I.cols - 1){
				cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i, j + 1))));
				g.add_edge(j + i*I.cols, j + 1 + i*I.cols, cost, cost);
			}
			if (i<I.rows - 1){
				cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i + 1, j))));
				g.add_edge(j + i*I.cols, j + I.cols*(i + 1), cost, cost);
				if (j<I.cols - 1){
					cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i + 1, j + 1))));
					g.add_edge(j + i*I.cols, j + 1 + I.cols*(i + 1), cost, cost);
				}
			}
		}
	}
	float flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols; j++){
			if (g.what_segment(j + i*I.cols) == Graph<float, float, float>::SINK){
				mask.at<uchar>(i, j) = 255;
			}
		}
	}
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols; j++){
			if (mask.at<uchar>(i, j) == 255){
				for (int k = -1; k < 2; k++){
					for (int l = -1; l < 2; l++){
						if (i + k >= 0 && i + k<I.rows && j + l >= 0 && j + l<I.cols){
							if (mask.at<uchar>(i + k, j + l) == 0){
								circle(I, Point(j, i), 1, Scalar(0, 0, 255));
							}
						}
					}
				}
			}
		}

	}
	//I.copyTo(I_fish, mask);
	imshow(window_name, I); waitKey();
}
*/

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
		cout << "Offset too great!" << endl;
		return;
	}
	// define infinity;
	double infinity = std::numeric_limits<double>::infinity();
	Point2i pt1, pt2;
	// compute overlap region;
	if (offset.y >= 0){
		const Point2i upperLeft(offset);
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
Mat synthesis(const Mat& img1, const Mat& img2, const Point2i& offset){
	Mat output;
	/*
	if (offset.y>=0){
	output = Mat::zeros(offset.y + img2.rows, img2.cols + offset.x, CV_8UC3);
	for (int i = 0; i < img1.rows; i++){
	for (int j = 0; j < img1.cols; j++){
	output.at<Vec3b>(i, j) = img1.at<Vec3b>(i, j);
	}
	}
	for (int i = offset.y; i < offset.y+img2.rows; i++){
	for (int j = offset.x; j < offset.x+img2.cols; j++){
	output.at<Vec3b>(i, j) = img2.at<Vec3b>(i-offset.y, j-offset.x);
	}
	}
	}
	else{
	output = Mat::zeros(-1 * offset.y + img1.rows, img2.cols + offset.x, CV_8UC3);
	for (int i = -offset.y; i < -offset.y+img1.rows; i++){
	for (int j = 0; j < img1.cols; j++){
	output.at<Vec3b>(i, j) = img1.at<Vec3b>(i+offset.y, j);
	}
	}
	for (int i = 0; i < img2.rows; i++){
	for (int j = offset.x; j < offset.x + img2.cols; j++){
	output.at<Vec3b>(i, j) = img2.at<Vec3b>(i, j - offset.x);
	}
	}
	}*/
	if (offset.x > 0){
		if (offset.y < 0){
			int height = abs(offset.y) + img1.rows > img2.rows ? abs(offset.y) + img1.rows : img2.rows;
			int width = abs(offset.x) + img2.cols > img1.cols ? abs(offset.x) + img2.cols : img1.cols;
			output = Mat::zeros(height, width, CV_8UC(img1.channels()));
			img1.copyTo(output(Rect(0, -offset.y, img1.cols, img1.rows)));
			img2.copyTo(output(Rect(offset.x, 0, img2.cols, img2.rows)));
		}
		else if (offset.y > 0){
			int height = abs(offset.y) + img2.rows > img1.rows ? abs(offset.y) + img2.rows : img1.rows;
			int width = abs(offset.x) + img2.cols > img1.cols ? abs(offset.x) + img2.cols : img1.cols;
			output = Mat::zeros(height, width, CV_8UC(img1.channels()));
			img1.copyTo(output(Rect(0, 0, img1.cols, img1.rows)));
			img2.copyTo(output(Rect(offset.x, offset.y, img2.cols, img2.rows)));
		}
	}
	graphCut(output, img1, img2, offset);
	return output;
}

int main() {
	//testGCuts();
	Mat img1, img2, img3, img4, img5, output;
	img1=imread("../image0006.jpg"); //CV_8UC3;
	img2 = imread("../image0007.jpg");
	img3 = imread("../image0008.jpg");
	img4 = imread("../IMG_0026.JPG");
	img5 = imread("../IMG_0027.JPG");
	/*
	imshow("I1", img1); 
	imshow("I2", img2);
	imshow("I3", img3);

	Point2i offset1(454, -8);
	output = synthesis(img1, img2, offset1);
	Point2i offset2(6, -314);
	output = synthesis(output, img3, offset2);
	*/
	Point2i offset(366, -9);
	output = synthesis(img4, img5, offset);
	imshow("Output", output);
	waitKey();
	return 0;
}
