#include "photomontage.hpp"
#include "graphCuts.hpp"
#include <cstdlib>
#include <queue>

using namespace std;
using namespace cv;
#define SIGGRAPH 0
#define PANORAMA 1

Mat panorama_(int argc, char **argv, int left, int right){
    if(right - left <= 1)
        return imread(argv[left]);
    int m = left + (right - left) / 2;
    Mat img1 = panorama_(argc, argv, left, m);
    //display("img1", img1);
    Mat img2 = panorama_(argc, argv, m, right);
    //display("img2", img2);
    Mat img2Regu;
    Point2i p = homoMatching(img1, img2, img2Regu);
    img2.release();
    //display("regulizaed", inputRegu);
    Mat output = showNaive(img1, img2Regu, p);
    img1.release();
    return output;
}
int panorama_(int argc, char **argv){
    if(argc < 2){
        cout<<"Usage: ./main img1 img2 ..."<<endl;
        return EXIT_FAILURE;
    }
    Mat output = panorama_(argc, argv, 1, argc);
    display("output", output);
    imwrite("output.jpg", output);
    output.release();
}
int panorama(int argc, char **argv){
    if(argc < 2){
        cout<<"Usage: ./main img1 img2 ..."<<endl;
        return EXIT_FAILURE;
    }
    Mat output = imread(argv[1]);
    if(output.empty()){
        cout<<"ERROR: could not open "<<argv[1]<<endl;
        return EXIT_FAILURE;
    }
    display(argv[1], output);
    for(int i = 2; i < argc; i++){
        Mat input = imread(argv[i]);
        if(input.empty()){
            cout<<"ERROR: could not open "<<argv[i]<<endl;
            break;
        }
        display(argv[i], input);
        Mat inputRegu;
        Point2i p = homoMatching(output, input, inputRegu);
        //display("regulizaed", inputRegu);
        cout<<p<<endl;
        output = showGraphCut(output, inputRegu, p);
        inputRegu.release();
    }
    display("output", output);
    imwrite("output.jpg", output);
}

int main(int argc, char **argv){
    panorama(argc, argv);
    /*Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    namedWindow( "img1", WINDOW_NORMAL);
    resizeWindow("img1", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img1", img1);
    namedWindow( "img2", WINDOW_NORMAL);
    resizeWindow("img2", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img2", img2);
    waitKey();

    set<Point2f> hist;*/
    /*Mat img2calib = relugarization(img1, img2);
    namedWindow( "img2calib", WINDOW_NORMAL);
    resizeWindow("img2calib", SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow("img2calib", img2calib);
    waitKey();

    pair<int, int> p = offset(img1, img2calib, hist, PANORAMA);
    cout<<p.first<<","<<p.second<<endl;
    //showNaive(img1, img2calib, p);


	Mat output =  showGraphCut(img1, img2calib, Point2i(p.first, p.second));
    namedWindow( "output", WINDOW_NORMAL);
    resizeWindow("output", SCREEN_WIDTH, SCREEN_HEIGHT);
	imshow("output", output);
	waitKey();
    imwrite("output.jpg", output);
    */
    /*Point2i p = offset(img1, img2, hist, SIGGRAPH);
    showNaive(img1, img2, p);
    Mat output = showGraphCut(img1, img2, p);
    namedWindow( "output", WINDOW_NORMAL);
    resizeWindow("output", SCREEN_WIDTH, SCREEN_HEIGHT);
	imshow("output", output);
	waitKey();*/
	return 0;
}

