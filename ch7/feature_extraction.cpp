#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if(argc != 3){
        cout<<"usage: feature-extraction img1 img2"<<endl;
        return 1;
    }
    // read data and initial 
    Mat m1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat m2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    std::vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptor1, descriptor2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // 1, detect the jiaodian
    orb->detect(m1, keypoint1);
    orb->detect(m2, keypoint2);

    // 2 
    orb->compute(m1, keypoint1, descriptor1);
    orb->compute(m2, keypoint2, descriptor2);

    Mat outimg1;
    drawKeypoints(m1, keypoint1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("orb", outimg1);

    // 3
    std::vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptor1, descriptor2, matches);

    // 4
    double min_dist = 10000, max_dist = 0;
    for(int i=0;i<descriptor1.rows; i++){
        double dist = matches[i].distance;
        if(dist > max_dist) max_dist = dist;
        if(dist < min_dist) min_dist = dist;
    }
    printf("max dist: %f\n", max_dist);
    printf("min dist: %f\n", min_dist);

    std::vector<DMatch> good_matches;
    for(int i=0; i<descriptor1.rows; i++){
        if(matches[i].distance < max(2*min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    // 5
    Mat img_match;
    Mat img_goodmatches;
    drawMatches(m1, keypoint1, m2, keypoint2, matches, img_match);
    drawMatches(m1, keypoint1, m2, keypoint2, good_matches, img_goodmatches);
    imshow("all", img_match);
    imshow("after filtering", img_goodmatches);
    waitKey(0);

    return 0;



}