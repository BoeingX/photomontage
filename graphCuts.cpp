#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

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

// calculate intensity of gradients Ix, Iy and G of image I;
void gradient(const Mat& I, Mat& grad_x, Mat& grad_y, Mat& grad){
	Mat tmp;
	Sobel(I, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(I, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	tmp = grad_x - grad_y;
	tmp = tmp.mul(tmp);
	sqrt(tmp, grad);
	convertScaleAbs(grad_x, grad_x);	//Scales, calculates absolute values, and converts the result to 8-bit;
	convertScaleAbs(grad_y, grad_y);	//Scales, calculates absolute values, and converts the result to 8-bit;
	convertScaleAbs(grad, grad);	//Scales, calculates absolute values, and converts the result to 8-bit;
}

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
			if (j<I.cols-1){
				cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i, j + 1))));
				g.add_edge(j + i*I.cols, j + 1 + i*I.cols, cost, cost);
			}
			if (i<I.rows-1){
				cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i + 1, j))));
				g.add_edge(j + i*I.cols, j + I.cols*(i + 1), cost, cost);
				if (j<I.cols-1){
					cost = 0.5*(GFun(float(grad.at<uchar>(i, j))) + GFun(float(grad.at<uchar>(i + 1, j + 1))));
					g.add_edge(j + i*I.cols, j + 1 + I.cols*(i + 1), cost, cost);
				}
			}
		}
	}
	float flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i = 0; i < I.rows; i++){
		for (int j = 0; j < I.cols;j++){
			if (g.what_segment(j + i*I.cols) == Graph<float, float, float>::SINK){
				mask.at<uchar>(i, j) = 255;
			}
		}
	}
	for (int i = 0; i < I.rows;i++){
		for (int j = 0; j < I.cols;j++){
			if (mask.at<uchar>(i, j)==255){
				for (int k = -1; k < 2;k++){
					for (int l = -1; l < 2;l++){
						if (i + k >= 0 && i+k<I.rows && j+l>=0 && j+l<I.cols){
							if (mask.at<uchar>(i+k, j+l)==0){
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

int main() {
	//testGCuts();
	Mat grad_x, grad_y, grad, gi, ge;
	Mat I=imread("../fishes.jpg"); //CV_8UC3;
	gi.create(I.size(), CV_32F);// create needed type;
	ge.create(I.size(), CV_32F);
	Vec3b algae(100, 80, 0); //BGR
	Vec3b fish(230, 211, 230);
	imshow("I",I);
	//setMouseCallback("I", onMouse1, &I);
	waitKey(0);
	gradient(I, grad_x, grad_y, grad);
	cost(I, gi, ge, fish, algae);	// pb: subtraction, and access to CV_64FC3
	imshow("gi", gi); 
	waitKey();
	imshow("ge", ge); 
	waitKey();
	GCuts(I, grad, gi, ge);
	return 0;
}
