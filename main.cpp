#include <cstdlib>
#include <queue>
#include "photomontage.hpp"
#include "graphCuts.hpp"

using namespace std;
using namespace cv;

int homo(int argc, char **argv){
    if(argc < 2){
        cout<<"Usage: ./main img1 img2 ..."<<endl;
        return EXIT_FAILURE;
    }
    Mat output = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(output.empty()){
        cout<<"ERROR: could not open "<<argv[1]<<endl;
        return EXIT_FAILURE;
    }
    display(argv[1], output);
    for(int i = 2; i < argc; i++){
        Mat input = imread(argv[i], CV_LOAD_IMAGE_COLOR);
        if(input.empty()){
            cout<<"ERROR: could not open "<<argv[i]<<endl;
            break;
        }
        display(argv[i], input);
        Mat inputRegu;
        Point2i p = homoMatching(output, input, inputRegu);
        cout<<p<<endl;
        output = showGraphCut(output, inputRegu, p);
        input.release();
    }
    display("output_homography", output);
    imwrite("output_homography.jpg", output);
    return EXIT_SUCCESS;
}

int entire(int argc, char **argv){
    if(argc < 2){
        cout<<"Usage: ./main img1 img2 ..."<<endl;
        return EXIT_FAILURE;
    }
    Mat output = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(output.empty()){
        cout<<"ERROR: could not open "<<argv[1]<<endl;
        return EXIT_FAILURE;
    }
    display(argv[1], output);
    for(int i = 2; i < argc; i++){
        Mat input = imread(argv[i], CV_LOAD_IMAGE_COLOR);
        if(input.empty()){
            cout<<"ERROR: could not open "<<argv[i]<<endl;
            break;
        }
        display(argv[i], input);
        Point2i p = entirePatchMatching(output, input);
        cout<<p<<endl;
        output = showGraphCut(output, input, p);
        input.release();
    }
    display("output_entire", output);
    imwrite("output_entire.jpg", output);
    return EXIT_SUCCESS;
}



int main(int argc, char **argv){
    homo(argc, argv);
    entire(argc, argv);
    return EXIT_SUCCESS;
}

