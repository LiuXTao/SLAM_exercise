#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

// 演示2D-2D的特征匹配估计相机运动

void find_feature_matches(
    const Mat& img1, const Mat& img2,
    std::vector<KeyPoint>& keypoint1,
    std::vector<KeyPoint>& keypoint2,
    std::vector<DMatch> &matches
);

void pose_estimation_2d2d(
    std::vector<KeyPoint> keypoint1,
    std::vector<KeyPoint> keyPoint2,
    std::vector<DMatch> matches,
    Mat &R, Mat &t
);
// 像素坐标 转 相机 归一化坐标
Point2d pixel2cam(const Point2d& p, const Mat&K);

int main(int argc, char **argv){
    if(argc != 3){
        cout<<"usage: pose estimation 2d2d img1 img2"<<endl;
        return 0;
    }
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoint1, keypoint2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoint1, keypoint2, matches);
    cout<<"totally find "<<matches.size() <<" pairs"<<endl;

    // estimate two pictures 
    Mat R, t;
    pose_estimation_2d2d(keypoint1, keypoint2, matches, R, t);

    Mat t_x = (Mat_<double> (3, 3) << 
        0, -t.at<double>(2, 0), t.at<double>(1, 0),
        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
        -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout<<"t^R = "<< endl << t_x*R << endl;

    // validate the dui ji yue shu
    Mat K = (Mat_<double> (3, 3) <<
        520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    
    for(DMatch m: matches){
        Point2d pt1 = pixel2cam(keypoint1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoint2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout<<"  epipolar constraint = " << d << endl;
    }
    return 0;

}

void find_feature_matches(    
    const Mat& img1, const Mat& img2,
    std::vector<KeyPoint>& keypoint1,
    std::vector<KeyPoint>& keypoint2,
    std::vector<DMatch> &matches){
    
    Mat descriptor1, descriptor2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );
    // 1
    detector->detect(img1, keypoint1);
    detector->detect(img2, keypoint2);
    // 2
    descriptor->compute(img1, keypoint1, descriptor1);
    descriptor->compute(img2, keypoint2, descriptor2);
    // 3
    vector<DMatch> match;
    matcher->match(descriptor1, descriptor2, match);
    // 4
    double min_dist = 10000, max_dist = 0;
    for(int i=0; i < descriptor1.rows; i++){
        int dist = match[i].distance;
        if(max_dist < dist) max_dist = dist;
        if(min_dist > dist) min_dist = dist;
    }
    printf("min distance: %f\n", min_dist);
    printf("max distance: %f\n", max_dist);

    for(int i=0; i < descriptor1.rows; i++){
        if(match[i].distance <= max(2 * min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d& p, const Mat& K )
{
    return Point2d(
        ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
        ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
    );
}

void pose_estimation_2d2d(
    std::vector<KeyPoint> keypoint1,
    std::vector<KeyPoint> keyPoint2,
    std::vector<DMatch> matches,
    Mat &R, Mat &t){
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> point1;
    vector<Point2f> point2;
    
    for(int i=0; i < (int) matches.size(); i ++){
        point1.push_back(keypoint1[matches[i].queryIdx].pt);
        point2.push_back(keyPoint2[matches[i].trainIdx].pt);
    }
    // f matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(point1, point2, CV_FM_8POINT);
    cout<<"fundamental matrix is "<< endl << fundamental_matrix << endl;

    // e matrix
    //相机光心, TUM dataset标定值
    Point2d principal_point(325.1, 249.7);
    //相机焦距, TUM dataset标定值
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(point1, point2, focal_length, principal_point, RANSAC);
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    // h matrix
    Mat homography_matrix;
    homography_matrix = findHomography(point1, point2, RANSAC, 3, noArray(), 2000, 0.99);
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    // recover pose information R, t
    recoverPose(essential_matrix, point1, point2, R, t, focal_length, principal_point);
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;

}

