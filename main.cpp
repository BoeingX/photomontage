#include "photomontage.hpp"
#include "graphCuts.hpp"
using namespace std;
using namespace cv;
#define SIGGRAPH 0
#define PANORAMA 1
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768

int main(int argc, char **argv){
    /*
    Mat img1 = read(argv[1]);
    Mat img2 = read(argv[2]);
    namedWindow( "img1", WINDOW_NORMAL);
    resizeWindow("img1", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img1", img1);
    namedWindow( "img2", WINDOW_NORMAL);
    resizeWindow("img2", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img2", img2);
    waitKey();

    set<Point2f> hist;
    Mat img2calib = calibration(img1, img2);
    namedWindow( "img2calib", WINDOW_NORMAL);
    resizeWindow("img2calib", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img2calib", img2calib);
    waitKey();
    pair<int, int> p = offset(img1, img2, hist, PANORAMA);
    showNaive(img1, img2calib, p);
    */

	Mat img1, img2, grad, gi, ge;
	Mat I=imread("fishes.jpg"); //CV_8UC3;
	//setMouseCallback("I", onMouse1, &I);
	gradient(I, img1, img2, grad);
	imshow("I1", img1); 
	imshow("I2", img2);
	Point2i offset(30, -30);
	//Point2i pt1, pt2;
	//correspondance(Point2i(90, 90), pt1, pt2, offset, img1, img2);
	//cout << pt1 << endl; cout << pt2 << endl;
	Mat output =  synthesis(img1, img2, offset);
	imshow("Output", output);
	waitKey();
	return 0;

    return 0;
}
