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

struct PnPProblem{
    PnPProblem(double x, double y, double X, double Y, double Z, double cx, double cy, double fx, double fy){
        x_ = x;
        y_ = y;
        X_ = X;
        Y_ = Y;
        Z_ = Z;
        cx_ = cx;
        cy_ = cy;
        fx_ = fx;
        fy_ = fy;
    }
    template<typename T>
    bool operator()(const T* const pose, T* residual) const{
        // 3D点存进p数据
        T p[3];
        p[0] = T(X_);
        p[1] = T(Y_);
        p[2] = T(Z_);
        // 相机位姿旋转
        T r[3];
        r[0] = pose[0];
        r[1] = pose[1];
        r[2] = pose[2];

        T newP[3];
        ceres::AngleAxisRotatePoint(r, p ,newP);

        // 3D point transition
        newP[0] += pose[3];
        newP[1] += pose[4];
        newP[2] += pose[5];

        T projectX = fx_ * newP[0] / newP[2] + cx_;
        T projectY = fy_ * newP[1] / newP[2] + cy_;

        residual[0] = T(x_) - projectX;
        residual[1] = T(y_) - projectY;
        return true;
    }

    double x_, y_;
    double X_, Y_, Z_;
    double cx_, cy_;
    double fx_, fy_;

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

void bundleAdjustmentCeres(const vector<Point3f> points_3d, const vector<Point2f> points_2d, const Mat& K, Mat &R, Mat &t, Mat& T){
    ceres::Problem problem;

    Mat rotateVector;
    // 调用罗格里斯公式
    Rodrigues(R, rotateVector);

    double pose[6];
    pose[0] = rotateVector.at<double>(0);
    pose[1] = rotateVector.at<double>(1);
    pose[2] = rotateVector.at<double>(2);
    pose[3] = t.at<double>(0);
    pose[4] = t.at<double>(1);
    pose[5] = t.at<double>(2);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for(size_t i = 0; i<points_3d.size(); i++){
        // 残差二维， 待优化参数六维， 不使用核函数
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PnPProblem, 2, 6>(new PnPProblem(
                points_2d[i].x, points_2d[i].y, 
                points_3d[i].x, points_3d[i].y, points_3d[i].z,
                cx, cy,
                fx, fy 
            )),
            nullptr,
            pose
        );
    }
    //  使用QR求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // 输出信息到std::cout
    options.minimizer_progress_to_stdout = true;

    // 开始优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<< summary.BriefReport()<<endl;

    // 输出坐标转化，由旋转向量转换为旋转矩阵
    rotateVector.at<double>(0) = pose[0];
    rotateVector.at<double>(1) = pose[1];
    rotateVector.at<double>(2) = pose[2];

    Rodrigues(rotateVector, R);

    t.at<double>(0) = pose[3];
    t.at<double>(1) = pose[4];
    t.at<double>(2) = pose[5];

    // 变换矩阵
    T = (Mat_<double>(4, 4)<<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2),
        0,                  0,                  0,                  1                
    );
    cout<<" after optimization \n";
    cout<< T <<endl;

}



int main ( int argc, char** argv ){
    if ( argc != 5 ){
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    // 定义3D和2D点
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d_1, pts_2d_2;

    for(DMatch m:matches){
        ushort d = d1.ptr<unsigned short> (int (keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        if(d == 0)
            continue;    
        float dd = d/5000.0;
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d_1.push_back(keypoints1[m.queryIdx].pt);
        pts_2d_2.push_back(keypoints2[m.trainIdx].pt);
    }
    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    Mat r, t;
    solvePnP(pts_3d, pts_2d_2, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    Mat R;
    cv::Rodrigues(r, R);

    cout<<" r = "<<endl << r <<endl;
    cout<<" R = "<<endl << R <<endl;
    cout<<" t = "<<endl << t <<endl;    

    cout<<"calling bundle adjustment"<<endl;
    // bundleAdjustment(pts_3d, pts_2d_2, K, R, t);
    Mat T;
    bundleAdjustmentCeres(pts_3d, pts_2d_2, K, R, t, T);
    
    return 0;
}


