#include "photomontage.hpp"
using namespace std;
using namespace cv;
int main(int argc, char **argv){
    Mat img1 = read(argv[1]);
    Mat img2 = read(argv[2]);
    /*imshow("img1", img1);
    imshow("img2", img2);
    waitKey(0);*/
    bestOffset(img1, img2);
    //bestOffset(img1, img2, 0.9, 0.9);
    return 0;
}
