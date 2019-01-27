#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include "ros/console.h"

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <iostream>
using namespace std;

const double PI = M_PI;

pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudaftDownSample(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudRadiusSearch(new pcl::PointCloud<pcl::PointXYZ>());
pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtreeForPriorMap(new pcl::search::KdTree<pcl::PointXYZ> ());

ros::Publisher pubLaserCloud_rawData, pubLaserCloud_aftSearch;

pcl::VoxelGrid<pcl::PointXYZ> downSizeFilter;

void laserCloudCallBack(const sensor_msgs::PointCloud2ConstPtr& laserCloudIn2)
{

      double time_start =  ros::Time::now().toSec();
      
      sensor_msgs::PointCloud2 laserRawData;
      laserRawData = * laserCloudIn2;
      laserRawData.header.frame_id = "/CYT";
      pubLaserCloud_rawData.publish(laserRawData);
      
      double rm_start =  ros::Time::now().toSec();
      pcl::fromROSMsg(*laserCloudIn2, *laserCloudIn);
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*laserCloudIn,*laserCloudIn, indices);
      int cloudSize = laserCloudIn->points.size();
      ROS_INFO("Size: %d", cloudSize);
      double rm_end =  ros::Time::now().toSec();
      ROS_INFO("Remove NaN Time Consume :%f", rm_end - rm_start);    
      
      double downSize_start =  ros::Time::now().toSec();
      downSizeFilter.setLeafSize(0.5, 0.5, 0.4); 
      downSizeFilter.setInputCloud(laserCloudIn);
      downSizeFilter.filter(*laserCloudaftDownSample);
      ROS_INFO("aft Down Sample Size: %lu", laserCloudaftDownSample->points.size());
      double downSize_end =  ros::Time::now().toSec();
      ROS_INFO("Down Size Filter Time Consume :%f", downSize_end - downSize_start);    

      kdtreeForPriorMap->setInputCloud(laserCloudaftDownSample);

     // Neighbors within radius search
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;
      pcl::PointXYZ searchPoint;

      searchPoint.x = 0.0;
      searchPoint.y = 0.0;
      searchPoint.z = 0.0;
      
      float radius = 50.0;
      //if ( 
      
      double search_start =  ros::Time::now().toSec();
      
      kdtreeForPriorMap->radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);// > 0 )

      double search_end =  ros::Time::now().toSec();
      ROS_INFO("Radius Search Time Consume :%f", search_end - search_start);    

          for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
              PointCloudRadiusSearch->points.push_back(laserCloudaftDownSample->points[pointIdxRadiusSearch[i]]);

    ROS_INFO ("aft Search Size: %lu", PointCloudRadiusSearch->points.size());
    sensor_msgs::PointCloud2 PCsearch;
    pcl::toROSMsg(*PointCloudRadiusSearch, PCsearch);
    PCsearch.header.frame_id = "/CYT";
    pubLaserCloud_aftSearch.publish(PCsearch);
    double time_end =  ros::Time::now().toSec();
    ROS_INFO("All Time Consume :%f", time_end - time_start);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "globalCorrection");
  ros::NodeHandle nh("~");

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/cloud_pcd", 10, laserCloudCallBack);
  
  pubLaserCloud_rawData = nh.advertise<sensor_msgs::PointCloud2> 
                                 ("/velodyne_cloud_rawdata", 2);
  
  pubLaserCloud_aftSearch = nh.advertise<sensor_msgs::PointCloud2> 
                                 ("/PtCloudSearch", 2);
  
  ros::spin();

  return 0;
}

