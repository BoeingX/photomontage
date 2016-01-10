#include "photomontage.hpp"
using namespace std;
using namespace cv;
#define SIGGRAPH 0
#define PANORAMA 1
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768
int main(int argc, char **argv){
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
    return 0;
}
