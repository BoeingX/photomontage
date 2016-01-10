#include "photomontage.hpp"
using namespace std;
using namespace cv;
#define SIGGRAPH 0
#define PANORAMA 1
int main(int argc, char **argv){
    Mat img1 = read(argv[1]);
    Mat img2 = read(argv[2]);
    imshow("img1", img1);
    imshow("img2", img2);
    waitKey();

    set<Point2f> hist;
    Mat img2calib = calibration(img1, img2);
    imshow("img2calib", img2calib);
    waitKey();

    pair<int, int> p = offset(img1, img2, hist, PANORAMA);
    cout<<p.first<<","<<p.second<<endl;
    showNaive(img1, img2calib, p);
    //pair<int, int> p = entirePatchMatching(img1, img2);
    //pair<int, int> p = entirePatchMatching(img1, img2, 0.9, 0.9);
    return 0;
}
