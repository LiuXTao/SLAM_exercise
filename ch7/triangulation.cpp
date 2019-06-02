#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

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

Point2f pixel2cam(const Point2d& p, const Mat& K )
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

void triangulation(
    const vector<KeyPoint>& keypoint1,
    const vector<KeyPoint>& keypoint2,
    const vector<DMatch>& matches,
    const Mat&R, const Mat& t,
    vector<Point3d>& points){
    
  Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m:matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( keypoint1[m.queryIdx].pt, K) );
        pts_2.push_back ( pixel2cam( keypoint2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}


int main(int argc, char **argv){
    if(argc != 3){
        cout<<"usage: triangulation 1.png 2.png"<<endl;
        return 0;
    }
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoint1, keypoint2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoint1, keypoint2, matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    Mat R, t;
    pose_estimation_2d2d(keypoint1, keypoint2, matches, R, t);

    // triangulation
    vector<Point3d> points;
    triangulation(keypoint1, keypoint2, matches, R, t, points);

    Mat K = (Mat_<double> (3, 3) <<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    for ( int i=0; i<matches.size(); i++ ){
        Point2d pt1_cam = pixel2cam( keypoint1[ matches[i].queryIdx ].pt, K );
        Point2d pt1_cam_3d(
            points[i].x/points[i].z, 
            points[i].y/points[i].z 
        );
        
        cout<<"point in the first camera frame: "<<pt1_cam<<endl;
        cout<<"point projected from 3D "<<pt1_cam_3d<<", d="<<points[i].z<<endl;
        
        // 第二个图
        Point2f pt2_cam = pixel2cam( keypoint2[ matches[i].trainIdx ].pt, K );
        Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z ) + t;
        pt2_trans /= pt2_trans.at<double>(2,0);
        cout<<"point in the second camera frame: "<<pt2_cam<<endl;
        cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
        cout<<endl;
    }
       return 1;
}

