#include "photomontage.hpp"
using namespace std;
using namespace cv;
Mat read(string filename){
    return imread(filename);
}

void write(const Mat &img, string filename){
    imwrite(filename, img);
}

void closure(Mat &img, const Mat &H){

}

void homography(Mat &img1, Mat &img2){
    const float inlier_threshold = 2.5f;
    const float nn_match_ratio = 0.8f;
    //! Points AKAZE and their descriptors
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    //! AKAZE method
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
    //! Match correspondent points
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    //! Calculate homography by RANSAC
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    
    vector<Point> obj, sce;
    for(int i = 0; i < matched1.size(); i++){
        obj.push_back(matched1[i].pt);
        sce.push_back(matched2[i].pt);
    }

    Mat homography = findHomography(obj, sce, CV_RANSAC);

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64FC1);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        float dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    int nbr_inliers = inliers1.size();
    Mat A(2*nbr_inliers, 9, CV_32FC1), h(9, nbr_inliers, CV_32FC1);
    for(int i = 0; i < nbr_inliers; i++){
        A.at<float>(2*i, 0) = -inliers1[i].pt.x;
        A.at<float>(2*i, 1) = -inliers1[i].pt.y;
        A.at<float>(2*i, 2) = -1.0f;
        A.at<float>(2*i, 3) = 0.0f;
        A.at<float>(2*i, 4) = 0.0f;
        A.at<float>(2*i, 5) = 0.0f;
        A.at<float>(2*i, 6) = inliers2[i].pt.x * inliers1[i].pt.x;
        A.at<float>(2*i, 7) = inliers2[i].pt.x * inliers1[i].pt.y;
        A.at<float>(2*i, 8) = inliers2[i].pt.x;
        A.at<float>(2*i+1, 0) = 0.0f;
        A.at<float>(2*i+1, 1) = 0.0f;
        A.at<float>(2*i+1, 2) = 0.0f;
        A.at<float>(2*i+1, 3) = -inliers1[i].pt.x;
        A.at<float>(2*i+1, 4) = -inliers1[i].pt.y;
        A.at<float>(2*i+1, 5) = -1.0f;
        A.at<float>(2*i+1, 6) = inliers2[i].pt.y * inliers1[i].pt.x;
        A.at<float>(2*i+1, 7) = inliers2[i].pt.y * inliers1[i].pt.y;
        A.at<float>(2*i+1, 8) = inliers2[i].pt.y;
    }
    SVD::solveZ(A, h); 
    
    //! Homography Matrix
    Mat H(3, 3, CV_32FC1);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            H.at<float>(i,j) = h.at<float>(i*3+j, 0);
    
    closure(img2, H); 
}
