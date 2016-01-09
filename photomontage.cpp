#include "photomontage.hpp"
#include <cmath>
#include <limits>
using namespace std;
using namespace cv;
Mat read(char *filename){
    Mat img = imread(filename);
    return img;
}

void write(const Mat &img, char *filename){
    imwrite(filename, img);
}

void closure(const Mat &img, const Mat &H, int &minX, int &maxX, int &minY, int &maxY){
    Mat H_inv;
    invert(H, H_inv);
    Mat xs(img.rows, img.cols, CV_32FC1);
    Mat ys(img.rows, img.cols, CV_32FC1);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            ys.at<float>(i,j) = (H_inv.at<float>(0,0)*j + H_inv.at<float>(0,1)*i + H_inv.at<float>(0,2)) / (H_inv.at<float>(2,0)*j + H_inv.at<float>(2,1)*i + H_inv.at<float>(2,2));
            xs.at<float>(i,j) = (H_inv.at<float>(1,0)*j + H_inv.at<float>(1,1)*i + H_inv.at<float>(1,2)) / (H_inv.at<float>(2,0)*j + H_inv.at<float>(2,1)*i + H_inv.at<float>(2,2));
        }
    }
    /*
    cout<<xs.at<float>(0,0)<<","<<ys.at<float>(0,0)<<endl;
    cout<<xs.at<float>(0,img.cols)<<","<<ys.at<float>(0,img.cols)<<endl;
    cout<<xs.at<float>(img.rows,0)<<","<<ys.at<float>(img.rows,0)<<endl;
    cout<<xs.at<float>(img.rows, img.cols)<<","<<ys.at<float>(img.rows, img.cols)<<endl;
    */
    double minX_, maxX_, minY_, maxY_;
    Point minLoc, maxLoc;
    minMaxLoc(xs, &minX_, &maxX_, &minLoc, &maxLoc, Mat());
    minMaxLoc(ys, &minY_, &maxY_, &minLoc, &maxLoc, Mat());
    minX = (int)(floor(minX_));
    maxX = (int)(ceil(maxX_));
    minY = (int)(floor(minY_));
    maxY = (int)(ceil(maxY_));
}

Mat homography(const Mat &img1, const Mat &img2){
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
    return H;
}
Mat glue(const Mat &img1, const Mat &img2){
    Mat H = homography(img1, img2);
    int minX, maxX, minY, maxY;    
    closure(img2, H, minX, maxX, minY, maxY); 
    Mat tmp1, tmp2; 
    copyMakeBorder(img1, tmp1, 0, 0, 0, img1.cols, BORDER_CONSTANT, 0);
    warpPerspective(img2, tmp2, H, Size(img1.cols, img1.rows), WARP_INVERSE_MAP) ;
    imshow("img1", tmp1);
    imshow("img2", tmp2);
    waitKey();

    Mat img = tmp1 + (tmp2 - tmp1);
    imshow("img1+img2", img);
    waitKey();
 
    return H;
}

pair<int, int> bestOffset(const Mat &img1, const Mat &img2, const float minPortionH, const float minPortionV){
    int xOpt, yOpt, x, y;
    double scoreOpt, score;
    scoreOpt = numeric_limits<double>::max();
    //! case 1
    bestOffsetTry(img1, img2, x, y, score, minPortionH, minPortionV); 
    if(score < scoreOpt){
        xOpt = x;
        yOpt = y;
        scoreOpt = score;
    }
    //! case 2
    bestOffsetTry(img2, img1, x, y, score, minPortionH, minPortionV);
    if(score < scoreOpt){
        xOpt = -x;
        yOpt = -y;
        scoreOpt = score;
    }
    Mat img1Flip, img2Flip;
    flip(img1, img1Flip, 1);
    flip(img2, img2Flip, 1);
    //! case 3
    bestOffsetTry(img1Flip, img2Flip, x, y, score, minPortionH, minPortionV);
    if(score < scoreOpt){
        xOpt = -x;
        yOpt = y;
    }
    //! case 4
    bestOffsetTry(img2Flip, img1Flip, x, y, score, minPortionH, minPortionV);
    if(score < scoreOpt){
       xOpt = x;
       yOpt = -y;
    }
    return make_pair<int, int>(xOpt, yOpt);
}

void bestOffsetTry(const Mat &img1, const Mat &img2, int &x, int &y, double &score, const float minPortionH, const float minPortionV){
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
    Mat tmp(img1.rows, img1.cols, CV_64FC1);
    score = numeric_limits<double>::max();
    float thresholdV = (1.0f - minPortionV)*img1.rows;
    float thresholdH = (1.0f - minPortionH)*img1.cols;
    for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++){
            if(i > thresholdV && j > thresholdH)
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
