#include<iostream>
#include<fstream>
#include<list>
#include<vector>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv){

    if(argc != 2){
        cout <<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_path = path_to_dataset + "/associate.txt";
    ifstream fin(associate_path);
    if( !fin ){
        cerr <<"cannot find associate.txt!"<<endl;
        return 1;
    }
    string rgb_file, depth_file, time_rgb, time_depth;
    // 使用list， 因为要删除跟踪失败的点
    list<cv::Point2f> keypoints;
    cv::Mat color, depth, last_color;

    for(int index = 0; index < 100; index++){
        // 从文件中读取图片信息， 并读入color, depth
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = imread(path_to_dataset + "/" + rgb_file);
        depth = imread(path_to_dataset + "/" + depth_file, -1);

        // 对第一帧提取 FAST 特征点
        if(index == 0){
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color, kps);
            for(auto kp: kps){
                keypoints.push_back(kp.pt);
            }
            last_color = color; 
            continue;
        }
        if(color.data == nullptr || depth.data == nullptr){
            continue;
        }
        // 对其他帧使用LK跟踪特征点
        vector<cv::Point2f> prev_keypoints, next_keypoints;
        for(auto kp: keypoints){
            prev_keypoints.push_back(kp);
        }
        vector<unsigned char> status;
        vector<float> errors;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, errors);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout<<"__ LK flow use time" << time_used.count() <<" s" <<endl;
        // 去除跟丢的点
        cout<< prev_keypoints.size() << endl;
        cout<< next_keypoints.size() << endl;

        int i = 0;
        for(auto iter = keypoints.begin(); iter!=keypoints.end(); i++){
            
            if(status[i] == 0){

                iter = keypoints.erase(iter);
                // cout << iter;
                continue;
            }
            *iter = next_keypoints[i];
            iter ++;

        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if(keypoints.size() == 0){
            cout<<" all keypoints are lost. "<<endl;
            break;
        }
        // 画出keypoints
        cv::Mat img_show = color.clone();
        for(auto kp: keypoints){
            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
        }
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }
    return 0;
}

