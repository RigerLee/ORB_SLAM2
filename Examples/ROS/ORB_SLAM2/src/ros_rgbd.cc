/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
    void PubPCL(const sensor_msgs::ImageConstPtr& msgRGB, const cv::Mat &imD, cv::Mat mTcw);

    ORB_SLAM2::System* mpSLAM;
    sensor_msgs::PointCloud dense_pcl;
};

ros::Publisher pub_dense_pcl;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    pub_dense_pcl = nh.advertise<sensor_msgs::PointCloud>("dense_pcl", 1000);

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat mTcw(4,4,CV_32F);
    mTcw = mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());

    PubPCL(msgRGB, cv_ptrD->image, mTcw);

}

cv::Mat coordinateTransform(cv::Mat mTcw)
{
  // rotate to world coordinates
  float rot[3][3] = {{0,-1,0},{0,0,-1},{1,0,0}};
  float trans[3]  = {0.,0.,0.5};
  cv::Mat mR1w = cv::Mat(3,3,CV_32F,rot);
  cv::Mat mtw1 = cv::Mat(3,1,CV_32F,trans);

  cv::Mat mRc1 = mTcw.rowRange(0,3).colRange(0,3);
  cv::Mat mtc1 = mTcw.rowRange(0,3).col(3);
  cv::Mat mt1c = -mRc1.t()*mtc1;
  cv::Mat mRcw = mRc1*mR1w;
  cv::Mat mtcw = -mRc1*mt1c - mRcw*mtw1;

  cv::Mat mTcwr = cv::Mat::eye(4,4,CV_32F);
  mRcw.copyTo(mTcwr.rowRange(0,3).colRange(0,3));
  mtcw.copyTo(mTcwr.rowRange(0,3).col(3));

  return mTcwr.clone();
}

void ImageGrabber::PubPCL(const sensor_msgs::ImageConstPtr& msgRGB, const cv::Mat &imD, cv::Mat mTcw)
{
    mTcw = coordinateTransform(mTcw);
    cv::Mat mRcw = mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat Rwc(3,3,CV_32FC1);
    Rwc = mRcw.t();
    cv::Mat mtcw = mTcw.rowRange(0,3).col(3);
    cv::Mat Ow(3,1,CV_32FC1);
    Ow = -Rwc*mtcw;

    int step = 10;
    //float fx = 505.876267, fy = 504.3448;
    //float cx = 312.2496, cy = 238.557867;
    float fx = 520.9, fy = 521.0;
    float cx = 325.1, cy = 249.7;
    float invfx = 1.0f/fx, invfy = 1.0f/fy;

    cv::Mat x3Dc = cv::Mat(3,1,CV_32F);

    for (int y = 0; y < imD.rows; y += step)
    {
      const uint16_t* depthPtr = imD.ptr<uint16_t>(y);

      for (int x = 0; x < imD.cols; x += step)
      {
        const uint16_t& d = depthPtr[x];
        //pcl::PointXYZRGB& pt = *pc_iter++;

        if ((d > 0) && (d < 30000))
        {
          const float u = x;//mMapx.at<float>(x,y);// not this remap matrix
          const float v = y;//mMapy.at<float>(x,y);
          const float xx = (u-cx)*d*invfx*0.0002;
          const float yy = (v-cy)*d*invfy*0.0002;
          x3Dc.at<float>(0,0) = xx;
          x3Dc.at<float>(1,0) = yy;
          x3Dc.at<float>(2,0) = d*0.0002;

          cv::Mat p =  Rwc*x3Dc+Ow;
          geometry_msgs::Point32 pts;
          pts.x = p.at<float>(0,0);
          pts.y = p.at<float>(1,0);
          pts.z = p.at<float>(2,0);
          dense_pcl.points.push_back(pts);
      }

    }
  }
  dense_pcl.header = msgRGB->header;
  pub_dense_pcl.publish(dense_pcl);
}
