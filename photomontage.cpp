#include "photomontage.hpp"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <algorithm>
#define SIGGRAPH 0
#define PANORAMA 1
#define MIN_OVERLAP 0.1
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768
#define INLINER_THRESHOLD 2.5
#define  NN_MATCH_RATIO  0.8
using namespace std;
using namespace cv;

void display(const char *name, Mat img, int showTime){
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, SCREEN_WIDTH, SCREEN_HEIGHT);
    imshow(name, img);
    waitKey(showTime);
}
void closure(const vector<Point2f> &pts, Point2f &ul, Point2f &lr){
    float minX, minY, maxX, maxY;
    minX = numeric_limits<float>::max();
    minY = numeric_limits<float>::max();
    maxX = -numeric_limits<float>::max();
    maxY = -numeric_limits<float>::max();
    for(int i = 0; i < pts.size(); i++){
        if(pts[i].x < minX)
            minX = pts[i].x;
        else if(pts[i].x > maxX)
            maxX = pts[i].x;
        if(pts[i].y < minY)
            minY = pts[i].y;
        else if(pts[i].y > maxY)
            maxY = pts[i].y;
    }
    ul = Point2f(minX, minY);
    lr = Point2f(maxX, maxY);
}

void inscribe(const vector<Point2f> &pts, Point2f &ul, Point2f &lr){
    vector<float> xs(4), ys(4);
    for(int i = 0; i < 4; i++){
        xs[i] = pts[i].x;
        ys[i] = pts[i].y;
    }
    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());
    ul = Point2f(xs[1], ys[1]);
    lr = Point2f(xs[2], ys[2]);
}

Mat homography(const Mat &img1, const Mat &img2){

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
        DMatch x = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < NN_MATCH_RATIO * dist2) {
            matched1.push_back(kpts1[x.queryIdx]);
            matched2.push_back(kpts2[x.trainIdx]);
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

        if(dist < INLINER_THRESHOLD) {
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

    return H;
}

Point2i homoMatching(const Mat &img1, const Mat &img2, Mat &img2Regu){
    Mat H = homography(img1, img2);
    Mat H_inv = H.inv();
    vector<Point2f> pts(4);
    pts[0] = Point2f(0, 0);
    pts[1] = Point2f(0, img2.rows);
    pts[2] = Point2f(img2.cols, 0);
    pts[3] = Point2f(img2.cols, img2.rows);
    vector<Point2f> dst(4);
    perspectiveTransform(pts, dst, H_inv);
    Point2f ul, lr;
    closure(dst, ul, lr);
    Mat M;
    M = (Mat_<float>(3, 3) << 1, 0, -ul.x, 0, 1, -ul.y, 0, 0, 1);
    Mat M_ = M * H_inv;
    Mat tmp;
    warpPerspective(img2, tmp, M_, Size(lr.x - ul.x, lr.y - ul.y));
    Point2f ul2, lr2;
    inscribe(dst, ul2, lr2);
    img2Regu = tmp(Rect(ul2.x - ul.x, ul2.y - ul.y, lr2.x - ul2.x, lr2.y - ul2.y));
    return Point2i(ul2.x, ul2.y);
}


Mat showNaive(const Mat &img1, const Mat &img2, const Point2i offset){
    Mat tmp1, tmp2;
    if(offset.x >= 0){
        if(offset.y < 0){
            int height = abs(offset.y) + img1.rows > img2.rows ? abs(offset.y) + img1.rows : img2.rows;
            int width = abs(offset.x) + img2.cols > img1.cols ? abs(offset.x) + img2.cols : img1.cols;
            tmp1 = Mat::zeros(height, width, CV_8UC(img1.channels()));
            img1.copyTo(tmp1(Rect(0, -offset.y, img1.cols, img1.rows)));
            tmp2 = Mat::zeros(height, width, CV_8UC(img1.channels()));
            img2.copyTo(tmp2(Rect(offset.x, 0, img2.cols, img2.rows)));
        }
        else if(offset.y >= 0){
            int height = abs(offset.y) + img2.rows > img1.rows ? abs(offset.y) + img2.rows : img1.rows;
            int width = abs(offset.x) + img2.cols > img1.cols ? abs(offset.x) + img2.cols : img1.cols;
            tmp1 = Mat::zeros(height, width, CV_8UC(img1.channels()));
            img1.copyTo(tmp1(Rect(0, 0, img1.cols, img1.rows)));
            tmp2 = Mat::zeros(height, width, CV_8UC(img1.channels()));
            img2.copyTo(tmp2(Rect(offset.x, offset.y, img2.cols, img2.rows)));
        }
    }
    Mat output = tmp1 + (tmp2 - tmp1);
    return output;
}

Point2i entirePatchMatching(const Mat &img1, const Mat &img2, set<Point2f> &hist){
    int xOpt, yOpt, x, y;
    double scoreOpt, score;
    scoreOpt = numeric_limits<double>::max();
    //! case 1
    entirePatchMatchingTry(img1, img2, x, y, score, hist);
    if(score < scoreOpt){
        xOpt = x;
        yOpt = y;
        scoreOpt = score;
    }
    //! case 2
    entirePatchMatchingTry(img2, img1, x, y, score, hist);
    if(score < scoreOpt){
        xOpt = -x;
        yOpt = -y;
        scoreOpt = score;
    }
    Mat img1Flip, img2Flip;
    flip(img1, img1Flip, 1);
    flip(img2, img2Flip, 1);
    //! case 3
    entirePatchMatchingTry(img1Flip, img2Flip, x, y, score, hist);
    if(score < scoreOpt){
        xOpt = -x;
        yOpt = y;
    }
    //! case 4
    entirePatchMatchingTry(img2Flip, img1Flip, x, y, score, hist);
    if(score < scoreOpt){
       xOpt = x;
       yOpt = -y;
    }
    
    return Point2i(xOpt, yOpt);
}

void entirePatchMatchingTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score, set<Point2f> &hist){
    Mat img1Gray, img2Gray;
    cvtColor(img1, img1Gray, CV_BGR2GRAY);
    cvtColor(img2, img2Gray, CV_BGR2GRAY);
    img1Gray.convertTo(img1Gray, CV_32F);
    img2Gray.convertTo(img2Gray, CV_32F);
    Mat img1Sum, img2Sum, img1Sum2, img2Sum2;
    integral(img1Gray, img1Sum, img1Sum2); 
    integral(img2Gray, img2Sum, img2Sum2); 
    Mat correlation;
    filter2D(img1Gray, correlation, CV_64FC1, img2Gray, Point(0, 0), 0, BORDER_CONSTANT); 
    score = numeric_limits<double>::max();
    for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++){
            float overlapH = j + img2.cols > img1.cols ? (img1.cols - j) / (float) img1.cols : img2.cols / (float) img1.cols;
            float overlapV = i + img2.rows > img1.rows ? (img1.rows - i) / (float) img1.rows : img2.rows / (float) img1.rows;
            if(overlapH < MIN_OVERLAP || overlapV < MIN_OVERLAP)
                continue;
            int s = i + img2.rows > img1.rows ? img1.rows : i + img2.rows;
            int t = j + img2.cols > img1.cols ? img1.cols : j + img2.cols;
            double A = img1Sum2.at<double>(s, t) - img1Sum2.at<double>(i, t) - img1Sum2.at<double>(s, j) + img1Sum2.at<double>(i, j);
            s = img2.rows + i > img1.rows ? img1.rows - i : img2.rows;
            t = img2.cols + j > img1.cols ? img1.cols - j : img2.cols;
            double B = img2Sum2.at<double>(s, t) - img2Sum2.at<double>(0, t) - img2Sum2.at<double>(s, 0) + img2Sum2.at<double>(0, 0);
            double C = A + B - 2.0 * correlation.at<double>(i, j);
            C /= (img1.rows - i)*(img1.cols - j);
            if(C < score){
                x = j;
                y = i;
                score = C;
            }
        }
    }
}
