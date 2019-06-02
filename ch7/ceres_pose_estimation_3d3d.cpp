#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>

#include<ceres/ceres.h>
#include<ceres/rotation.h>
#include<chrono>

using namespace std;
using namespace cv;

struct ICPProblem{
    ICPProblem(double x1, double y1, double z1, double x2, double y2, double z2){
        x1_ = x1;
        x2_ = x2;

        y1_ = y1;
        y2_ = y2;

        z1_ = z1;
        z2_ = z2;
    }
    template<typename T>
    bool operator()(const T* const pose, T* residual) const{
        T p[3];
        p[0] = T(x2_);
        p[1] = T(y2_);
        p[2] = T(z2_);

        T r[3];
        r[0] = pose[0];
        r[1] = pose[1];
        r[2] = pose[2];

        T newP[3];
        ceres::AngleAxisRotatePoint(r, p, newP);

        newP[0] += pose[3];
        newP[1] += pose[4];
        newP[2] += pose[5];

        residual[0] = T(x1_) - newP[0];
        residual[1] = T(y1_) - newP[1];
        residual[2] = T(z1_) - newP[2];
        return true; 
    }


    double x1_, y1_, z1_;
    double x2_, y2_, z2_; 
};

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches ){
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matchers;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matchers );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matchers[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matchers[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( matchers[i] );
        }
    }
}


// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K ){
    return Point2d(
            ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
            ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
        );

}

void pose_estimation_3d3d(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t){
    Point3f p1, p2;
    int N = pts1.size();
    for(int i = 0; i < N; i++){
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= N;
    p2 /= N;

    vector<Point3f> q1(N), q2(N);
    for(int i=0; i < N; i++){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;

    }
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i=0; i < N; i++){
        // W = q1*q2^T
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout<<" W = "<<endl<<W<<endl;

    // SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if(U.determinant() * V.determinant() < 0){
        for(int x = 0; x < 3; x++){
            U(x, 2) *= -1;
        }
    }    
    cout<<"U = "<<endl<<U<<endl;
    cout<<"V = "<<endl<<V<<endl;

    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    R = (Mat_<double>(3, 3)<<    
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1)<<
        t_(0, 0), t_(1, 0), t_(2, 0)
    );
    cout<<" after SVD,\n R:"<<endl<<R<<endl<<"t:"<<endl<<t<<endl;
}

void bundleAdjustmentCeres(const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat &R, Mat &t, Mat &T){
    ceres::Problem problem;

    Mat rotateVector;
    Rodrigues(R, rotateVector);

    double pose[6];
    pose[0] = rotateVector.at<double>(0);
    pose[1] = rotateVector.at<double>(1);
    pose[2] = rotateVector.at<double>(2);
    pose[3] = t.at<double>(0);
    pose[4] = t.at<double>(1);
    pose[5] = t.at<double>(2);

    for(size_t i = 0; i < pts1.size(); i++){
        // 残差二维，待优化参数六维， 不使用核函数
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ICPProblem, 3, 6>(new ICPProblem(
                pts1[i].x, pts1[i].y, pts1[i].z,
                pts2[i].x, pts2[i].y, pts2[i].z
            )),
            nullptr,
            pose
        );
    }
    
    // QR
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    rotateVector.at<double>(0) = pose[0];
    rotateVector.at<double>(1) = pose[1];
    rotateVector.at<double>(2) = pose[2];
    t.at<double>(0) = pose[3];
    t.at<double>(1) = pose[4];
    t.at<double>(2) = pose[5];

    Rodrigues(rotateVector, R);

    T = (Mat_<double>(4, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2),
        0,                  0,                  0,                  1
    );
    cout<<" after optimaze" << endl<<T<<endl;
}


int main(int argc, char** argv){
    if ( argc != 5 ){
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    Mat img_dep1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat img_dep2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    
    for(DMatch m: matches){
        ushort d1 = img_dep1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        ushort d2 = img_dep2.ptr<unsigned short>(int(keypoints2[m.trainIdx].pt.y))[int(keypoints2[m.trainIdx].pt.x)];

        if(d1 == 0 || d2 == 0)
            continue;
        
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints2[m.trainIdx].pt, K);

        float dd1 = float(d1) / 1000.0;
        float dd2 = float(d2) / 1000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));

    }
    cout<<"3d - 3d paris: "<< pts1.size()<<endl;
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout<<" R_inv = "<<endl<<R.t()<<endl;
    cout<<" t_inv = "<<endl<<-R.t()<<endl;

    Mat T;
    bundleAdjustmentCeres(pts1, pts2, R, t, T);

    return 0;    
}