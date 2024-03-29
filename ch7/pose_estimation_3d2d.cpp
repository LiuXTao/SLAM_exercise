#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
#include<g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include<chrono>


using namespace std;
using namespace cv;

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
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
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

void bundleAdjustment(const vector<Point3f> points_3d, const vector<Point2f> points_2d, const Mat& K, Mat &R, Mat &t){
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType> (); // 线性求解器
    Block* solver_ptr = new Block(linearSolver);  // 矩阵求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // 定义梯度优化策略   
    g2o::SparseOptimizer optimizer;  // 定义优化器
    optimizer.setAlgorithm(solver);  //

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId( 0 );
    pose->setEstimate(g2o::SE3Quat(
        R_mat, Eigen::Vector3d( t.at<double>(0, 0), t.at<double> (1, 0), t.at<double>(2, 0))
    ));
    optimizer.addVertex( pose );

    // landmarks
    int index = 1;
    for(const Point3f p: points_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);   
    }
    
    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );

    camera->setId(0);
    optimizer.addParameter(camera);

//  edge
    index = 1;
    for(const Point2f p: points_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y ));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());        
        optimizer.addEdge(edge);
        index ++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout<<"optimization costs time:"<<time_used.count() <<" s"<<endl;

    cout<<endl<<"after optimization: "<<endl;
    cout<<" T = "<<endl<<Eigen::Isometry3d(pose->estimate()).matrix()<<endl;    
}

void bundleAdjustment_with_first_camera(
    const vector<Point3f> pts_3d, const vector<Point2f> pts_2d_1, const vector<Point2f> pts_2d_2, const Mat& K, Mat& R, Mat& t
){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> > Block;
    // 设置线性求解器， 使用CSparse分解
    Block::LinearSolverType *linear_solver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    // 矩阵求解器
    Block* solver_ptr = new Block(linear_solver);

    // 设置levenberg梯度下降法求解  
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // 定义优化器
    g2o::SparseOptimizer optimizer;
    // 设置优化器求解算法
    optimizer.setAlgorithm(solver);

    // first camera 位姿 (加入第一个相机的位姿)
    g2o::VertexSE3Expmap* poseOne = new g2o::VertexSE3Expmap();
    poseOne->setId(0);
    poseOne->setFixed(1);
    poseOne->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(poseOne);

    // second camera  位姿  (必加)
    g2o::VertexSE3Expmap* poseTwo = new g2o::VertexSE3Expmap();
    poseTwo->setId(1);
    Eigen::Matrix3d R_two;
    R_two <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    // 设置待优化参数为旋转矩阵和评议矩阵
    poseTwo->setEstimate(g2o::SE3Quat(
        R_two,
        Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))
    ));
    optimizer.addVertex(poseTwo);

    int index = 2;
    for(const Point3f p:pts_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        // 待优化空间点的3D位置
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        // 是否边缘化进行稀疏求解
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }
    // 对相机内参进行优化
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double> (0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);
    // 添加边
    int edgecount = 0;
    // 第一个相机观测
    index = 2;
    for(const Point2f p : pts_2d_1){
        // 重投影误差边类
        g2o::EdgeProjectXYZ2UV* edgeOne = new g2o::EdgeProjectXYZ2UV();
        edgeOne->setId(edgecount++);
        // 链接两个顶点
        edgeOne->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edgeOne->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));
        // 测量值位第一帧的像素坐标
        edgeOne->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edgeOne->setParameterId(0, 0);
        edgeOne->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edgeOne);
        index++;
    }
    // 第二个相机观测
    index = 2;
    for(const Point2f p:pts_2d_2){
        g2o::EdgeProjectXYZ2UV* edgeTwo = new g2o::EdgeProjectXYZ2UV();
        edgeTwo->setId(edgecount++);
        edgeTwo->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edgeTwo->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)));
        edgeTwo->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edgeTwo->setParameterId(0, 0);
        edgeTwo->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edgeTwo);
        index++;
    }
    // 优化器初始化
    optimizer.initializeOptimization();
    // 设置优化器
    optimizer.optimize(100);
    cout<<"优化后"<<endl;
    cout<<"T1 = " << endl << Eigen::Isometry3d(poseOne->estimate()).matrix() << endl;
    cout<<"T2 = " << endl << Eigen::Isometry3d(poseTwo->estimate()).matrix() << endl;
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
    bundleAdjustment_with_first_camera(pts_3d, pts_2d_1, pts_2d_2, K, R, t);
    
    return 0;
}


