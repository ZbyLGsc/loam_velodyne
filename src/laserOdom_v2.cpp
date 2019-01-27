#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

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
//#include <pcl/point_representation.h>

#include <Eigen/Eigen>
using namespace Eigen;
using  std::cout;
using  std::endl;

Eigen::Vector3f R_to_rpy(const Eigen::Matrix3f& R)
{
  Eigen::Vector3f rpy;
  float r = asin(R(2,1));
  float p = atan2(-R(2,0)/cos(r), R(2,2)/cos(r));
  float y = atan2(-R(0,1)/cos(r), R(1,1)/cos(r));
  rpy(0) = r;
  rpy(1) = p;
  rpy(2) = y;

  return rpy;
}

const double PI = M_PI;//3.1415926;

const float scanPeriod = 0.05;

const int skipFrameNum = 0;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;

pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsFlat(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr pointSearchCornerLast(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr pointSearchSurfLast(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr pointProjCornerLast(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr pointProjSurfLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr coeffSel(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZ>::Ptr imuTrans(new pcl::PointCloud<pcl::PointXYZ>());
//pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
//pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::search::KdTree<pcl::PointXYZI> ());
pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::search::KdTree<pcl::PointXYZI> ());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

int pointSelCornerInd[40000];
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

int pointSelSurfInd[40000];
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];

float transform[6] = {0};
float transformSum[6] = {0};

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;

//bool change_KF = true;

void TransformToStart(pcl::PointXYZI *pi, pcl::PointXYZI *po)
{
  
  if(std::isnan(pi->x)||std::isnan(pi->y)||std::isnan(pi->z))
  {
    ROS_WARN("##########################################");
  	std::cout<<"pi.x: "<<pi->x<<"pi.y: "<<pi->y<<"pi.z: "<<pi->z<<std::endl;
    ROS_BREAK();
  }
  
  float s =1;// 20 * (pi->intensity - int(pi->intensity));

  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  ROS_WARN(" In TransformToStart ");
  std::cout<<transform[0]<<std::endl;
  std::cout<<transform[1]<<std::endl;
  std::cout<<transform[2]<<std::endl;
  std::cout<<transform[3]<<std::endl;
  std::cout<<transform[4]<<std::endl;
  std::cout<<transform[5]<<std::endl;

  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;

  if(std::isnan(po->x)||std::isnan(po->y)||std::isnan(po->z))
  {
    ROS_WARN("##########################################");
    std::cout<<"s: "<<s<<std::endl;
  std::cout<<transform[0]<<std::endl;
  std::cout<<transform[1]<<std::endl;
  std::cout<<transform[2]<<std::endl;
  std::cout<<transform[3]<<std::endl;
  std::cout<<transform[4]<<std::endl;
  std::cout<<transform[5]<<std::endl;
  std::cout<<"rx :"<<rx<<"ry :"<<ry<<"rz: "<<rz<<"tx: "<<tx<<"ty: "<<ty<<"tz: "<<tz<<std::endl;
  std::cout<<"x1: "<<x1<<"y1: "<<y1<<"z1: "<<z1<<"x2: "<<x2<<"y2: "<<y2<<"z2: "<<z2<<std::endl;
  std::cout<<"po.x: "<<po->x<<"po.y: "<<po->y<<"po.z: "<<po->z<<std::endl;
    ROS_BREAK();
  }
  po->intensity = pi->intensity;
    
}

void TransformToEnd(pcl::PointXYZI *pi, pcl::PointXYZI *po)
{
  float s =1;// 20 * (pi->intensity - int(pi->intensity));//10

  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;

  rx = transform[0];
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  float x7 = cos(imuRollStart) * (x6 - imuShiftFromStartX) 
           - sin(imuRollStart) * (y6 - imuShiftFromStartY);
  float y7 = sin(imuRollStart) * (x6 - imuShiftFromStartX) 
           + cos(imuRollStart) * (y6 - imuShiftFromStartY);
  float z7 = z6 - imuShiftFromStartZ;

  float x8 = x7;
  float y8 = cos(imuPitchStart) * y7 - sin(imuPitchStart) * z7;
  float z8 = sin(imuPitchStart) * y7 + cos(imuPitchStart) * z7;

  float x9 = cos(imuYawStart) * x8 + sin(imuYawStart) * z8;
  float y9 = y8;
  float z9 = -sin(imuYawStart) * x8 + cos(imuYawStart) * z8;

  float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
  float y10 = y9;
  float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

  float x11 = x10;
  float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
  float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

  po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
  po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
  po->z = z11;
  po->intensity = int(pi->intensity);
}

void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                       float alx, float aly, float alz, float &acx, float &acy, float &acz)
{
  float sbcx = sin(bcx);
  float cbcx = cos(bcx);
  float sbcy = sin(bcy);
  float cbcy = cos(bcy);
  float sbcz = sin(bcz);
  float cbcz = cos(bcz);

  float sblx = sin(blx);
  float cblx = cos(blx);
  float sbly = sin(bly);
  float cbly = cos(bly);
  float sblz = sin(blz);
  float cblz = cos(blz);

  float salx = sin(alx);
  float calx = cos(alx);
  float saly = sin(aly);
  float caly = cos(aly);
  float salz = sin(alz);
  float calz = cos(alz);

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
            - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
            - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  acx = -asin(srx);

  float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
  
  float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
               - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
               - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
               + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
               - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
               + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
               + calx*cblx*salz*sblz);
  float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
               - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
               + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
               + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
               + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
               - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
               - calx*calz*cblx*sblz);
  acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
}

void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                        float &ox, float &oy, float &oz)
{
  float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
  ox = -asin(srx);

  float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
               + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
  float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
               - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
               + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
  float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
               - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

    cornerPointsSharp->clear();
    pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
    newCornerPointsSharp = true;

}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

      cornerPointsLessSharp->clear();
      pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cornerPointsLessSharp,*cornerPointsLessSharp, indices);
      newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

      surfPointsFlat->clear();
      pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*surfPointsFlat,*surfPointsFlat, indices);
      newSurfPointsFlat = true;

}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

      surfPointsLessFlat->clear();
      pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*surfPointsLessFlat,*surfPointsLessFlat, indices);
      newSurfPointsLessFlat = true;

}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes,*laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}

void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTrans2)
{
  timeImuTrans = imuTrans2->header.stamp.toSec();

  imuTrans->clear();
  pcl::fromROSMsg(*imuTrans2, *imuTrans);
/*
  imuPitchStart = imuTrans->points[0].x;
  imuYawStart = imuTrans->points[0].y;
  imuRollStart = imuTrans->points[0].z;

  imuPitchLast = imuTrans->points[1].x;
  imuYawLast = imuTrans->points[1].y;
  imuRollLast = imuTrans->points[1].z;
*/
  imuRollStart = imuTrans->points[0].x;
  imuPitchStart = imuTrans->points[0].y;
  imuYawStart = imuTrans->points[0].z;

  imuRollLast = imuTrans->points[1].x;
  imuPitchLast = imuTrans->points[1].y;
  imuYawLast = imuTrans->points[1].z;

  imuShiftFromStartX = imuTrans->points[2].x;
  imuShiftFromStartY = imuTrans->points[2].y;
  imuShiftFromStartZ = imuTrans->points[2].z;

  imuVeloFromStartX = imuTrans->points[3].x;
  imuVeloFromStartY = imuTrans->points[3].y;
  imuVeloFromStartZ = imuTrans->points[3].z;

  newImuTrans = true;
}
bool vicon_init = true;
double origin_x, origin_y, origin_z;
Eigen::Quaterniond origin_q;
Eigen::Matrix3d origin_R;
ros::Publisher pub_vicon_odom;
ros::Publisher pub_vicon_path;
nav_msgs::Path path_vicon;

void viconCallBack(const geometry_msgs::PoseStamped::ConstPtr &vicon_odom )
{
    if(vicon_init)
    {
        origin_x = -vicon_odom->pose.position.y;
        origin_y = vicon_odom->pose.position.x;
        origin_z = vicon_odom->pose.position.z;
        origin_q.x() = vicon_odom->pose.orientation.x;
        origin_q.y() = vicon_odom->pose.orientation.y;
        origin_q.z() = vicon_odom->pose.orientation.z;
        origin_q.w() = vicon_odom->pose.orientation.w;
        origin_R = origin_q;

        vicon_init=false;
        return;
    }

    nav_msgs::Odometry vicon_odom_;
    vicon_odom_.header = vicon_odom->header;
    vicon_odom_.header.frame_id = "/map";
    vicon_odom_.pose.pose.position.x = -vicon_odom->pose.position.y - origin_x;
    vicon_odom_.pose.pose.position.y = vicon_odom->pose.position.x - origin_y;
    vicon_odom_.pose.pose.position.z = vicon_odom->pose.position.z - origin_z;
    vicon_odom_.pose.pose.orientation.w = vicon_odom->pose.orientation.w;
    vicon_odom_.pose.pose.orientation.x = vicon_odom->pose.orientation.x;
    vicon_odom_.pose.pose.orientation.y = vicon_odom->pose.orientation.y;
    vicon_odom_.pose.pose.orientation.z = vicon_odom->pose.orientation.z;
    
    path_vicon.header.frame_id = "/map";
    path_vicon.header.stamp = vicon_odom_.header.stamp;
    
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = vicon_odom_.pose.pose.position.x;
    pose.pose.position.y = vicon_odom_.pose.pose.position.y;
    pose.pose.position.z = vicon_odom_.pose.pose.position.z;
    pose.pose.orientation.w = vicon_odom_.pose.pose.orientation.w;
    pose.pose.orientation.x = vicon_odom_.pose.pose.orientation.x;
    pose.pose.orientation.y = vicon_odom_.pose.pose.orientation.y;
    pose.pose.orientation.z = vicon_odom_.pose.pose.orientation.z;
    pose.header = path_vicon.header;
    path_vicon.poses.push_back(pose);
    
    pub_vicon_path.publish(path_vicon); 
    pub_vicon_odom.publish(vicon_odom_);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;

  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_sharp", 1, laserCloudSharpHandler);

  ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                             ("/laser_cloud_less_sharp", 1, laserCloudLessSharpHandler);

  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                      ("/laser_cloud_flat", 1, laserCloudFlatHandler);

  ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                          ("/laser_cloud_less_flat", 1, laserCloudLessFlatHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2> 
                                         ("/velodyne_cloud_2", 1, laserCloudFullResHandler);
  //ros::Subscriber subVicon = nh.subscribe<geometry_msgs::PoseStamped>
                                      //   ("/Robot_1/pose",10, viconCallBack);                                         
  //ros::Subscriber subImuTrans = nh.subscribe<sensor_msgs::PointCloud2> 
  //                              ("/imu_trans", 5, imuTransHandler);

  ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_corner_last", 1);

  ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_surf_last", 1);

  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2> 
                                        ("/velodyne_cloud_3", 1);

  pub_vicon_odom = nh.advertise<nav_msgs::Odometry>("/vicon_odom", 10);
  pub_vicon_path = nh.advertise<nav_msgs::Path>("/vicon_path",10);
  //ros::Publisher pub1 = nh.advertise<sensor_msgs::PointCloud2> ("/pc1", 2);

  //ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2> ("/pc2", 2);

  //ros::Publisher pub3 = nh.advertise<sensor_msgs::PointCloud2> ("/pc3", 2);

  //ros::Publisher pub4 = nh.advertise<sensor_msgs::PointCloud2> ("/pc4", 2);

  ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 10);
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/map";
  //laserOdometry.child_frame_id = "/laser_odom";

  //tf::TransformBroadcaster tfBroadcaster;
  //tf::StampedTransform laserOdometryTrans;
  //laserOdometryTrans.frame_id_ = "/camera_init";
  //laserOdometryTrans.child_frame_id_ = "/laser_odom";

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  pcl::PointXYZI pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

  bool isDegenerate = false;
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  int frameCount = skipFrameNum;
  ros::Rate rate(20);// should be 50 ???
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

//newImuTrans = true;

   //change_KF = false;
  /*&&
        fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.01 &&
        fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.01 &&
        fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.01 &&
        fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.01 */
    if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat && 
        newSurfPointsLessFlat && newLaserCloudFullRes)// && newImuTrans )//&&
        //fabs(timeImuTrans - timeSurfPointsLessFlat) < 0.005) {
        {

      ros::Time time_start = ros::Time::now();
      newCornerPointsSharp = false;
      newCornerPointsLessSharp = false;
      newSurfPointsFlat = false;
      newSurfPointsLessFlat = false;
      newLaserCloudFullRes = false;
      newImuTrans = false;

      if (!systemInited) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
        std::vector<int> indices_2;
        pcl::removeNaNFromPointCloud(*laserCloudSurfLast,*laserCloudSurfLast, indices_2);

        ROS_WARN("[Laser Odom Node] Initialize ? ... ");
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast); //sharp point last
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast); //flat point last

        //ROS_WARN("check cloud output size ...");
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);


        //transformSum[0] += imuRollStart;
        //transformSum[1] += imuYawStart;// comment original
        //transformSum[1] += imuPitchStart;

        systemInited = true;
        continue;
      }

      laserCloudOri->clear();
      coeffSel->clear();

      //transform[0] = 0.0;
      //transform[1] = 0.0;
      //transform[2] = 0.0;
      //transform[3] -= imuVeloFromStartX * scanPeriod;
      //transform[4] -= imuVeloFromStartY * scanPeriod;
      //transform[5] -= imuVeloFromStartZ * scanPeriod;

      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {//10 100
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
        std::vector<int> indices_2;
        pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices_2);

        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();

        float deltaR = 0.0, deltaT = 0.0;
        int iterCount;

        for (iterCount = 0; iterCount < 40; iterCount ++) // 25
        {
          for (int i = 0; i < cornerPointsSharpNum; i++) 
          {
            ROS_WARN("check corner points");
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);
            if (1)//(iterCount % 5 == 0) 
            {
              //if(isnan(pointSel.x)||isnan(pointSel.y)||isnan(pointSel.z))
              //    continue;
              //std::vector<int> indices;
              //pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
              kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // 1
              int closestPointInd = -1, minPointInd2 = -1;
              if (pointSearchSqDis[0] < 25) 
              {//25
                closestPointInd = pointSearchInd[0];
                int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

              //ROS_WARN("check nearst corner index ... ");
              //cout<<closestPointInd<<endl;
              //cout<<minPointInd2<<endl;
               
                float pointSqDis, minPointSqDis2 = 25;//25 // ### Indoor 10 is perfect ... 
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) 
                {
                      if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) //Original: 2.5 ## Indoor 1.5 is perfect ...
                           break;//break

                      pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                   (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                   (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                   (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                   (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                   (laserCloudCornerLast->points[j].z - pointSel.z);

                      if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) // >
                      {
                            if (pointSqDis < minPointSqDis2) 
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                      }
                }
              
             // cout<<closestPointInd<<endl;
             // cout<<minPointInd2<<endl;

                for (int j = closestPointInd - 1; j >= 0; j--) {
                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;//break// continue?
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);

                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
              }

             // cout<<closestPointInd<<endl;
              //cout<<minPointInd2<<endl;

              pointSearchCornerInd1[i] = closestPointInd;
              pointSearchCornerInd2[i] = minPointInd2;
            }

            if (pointSearchCornerInd2[i] >= 0) { // Not consistent
              tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
              tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

              float x0 = pointSel.x;
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              float x1 = tripod1.x;
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              float x2 = tripod2.x;
              float y2 = tripod2.y;
              float z2 = tripod2.z;

              float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                         * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                         + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                         * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                         + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                         * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

              float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

              float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

              float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                       + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              float ld2 = a012 / l12;

              pointProj = pointSel;
              pointProj.x -= la * ld2;
              pointProj.y -= lb * ld2;
              pointProj.z -= lc * ld2;

              float s = 1;
               if(iterCount >= 1)//(iterCount >= 5) /iterCount >= 1
                s = 1 - 1.8 * fabs(ld2);
              
              
              coeff.x = s * la;
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

              if (s > 0.1 && ld2 != 0) {
                laserCloudOri->push_back(cornerPointsSharp->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }

          for (int i = 0; i < surfPointsFlatNum; i++) 
          {
            ROS_WARN("check surface points");
            TransformToStart(&surfPointsFlat->points[i], &pointSel);
            if (1)//(iterCount % 5 == 0) {//iterCount % 5 == 0
            {
              kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // 1
              int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
              if (pointSearchSqDis[0] < 25) //#####: Indoor 10 is perfect ..
              {
                  closestPointInd = pointSearchInd[0];
                  int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);
                  float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;

                  for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) 
                  {
                    if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) 
                      break;

                    pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                 (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                 (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                 (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                 (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                 (laserCloudSurfLast->points[j].z - pointSel.z);

                    if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) 
                    {
                           if (pointSqDis < minPointSqDis2) 
                           {
                             minPointSqDis2 = pointSqDis;
                             minPointInd2 = j;
                           }
                    } 
                    else 
                    {
                           if (pointSqDis < minPointSqDis3) 
                           {
                             minPointSqDis3 = pointSqDis;
                             minPointInd3 = j;
                           }
                    }
                  }
                    
                  for (int j = closestPointInd - 1; j >= 0; j--) 
                  {
                    if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) 
                       break;

                    pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                 (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                 (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                 (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                 (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                 (laserCloudSurfLast->points[j].z - pointSel.z);

                    if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) 
                    {
                          if (pointSqDis < minPointSqDis2) 
                          {
                              minPointSqDis2 = pointSqDis;
                              minPointInd2 = j;
                          }
                    } 
                    else 
                    {
                          if (pointSqDis < minPointSqDis3) 
                          {
                               minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                          }
                    }
                  }
                }
                    
              //cout<<closestPointInd<<endl;
              //cout<<minPointInd2<<endl;
              //cout<<minPointInd3<<endl;

              pointSearchSurfInd1[i] = closestPointInd;
              pointSearchSurfInd2[i] = minPointInd2;
              pointSearchSurfInd3[i] = minPointInd3;
            }

            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
              tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
              tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
              tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                       - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                       - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                       - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

              float ps = sqrt(pa * pa + pb * pb + pc * pc);
              pa /= ps;
              pb /= ps;
              pc /= ps;
              pd /= ps;

              float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

              pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;

              float s = 1;
              if(iterCount >= 1)//(iterCount >= 5)
                s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
              

              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) {
                laserCloudOri->push_back(surfPointsFlat->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }

          int pointSelNum = laserCloudOri->points.size();
          ROS_ERROR("check pointSelNum : %d", pointSelNum);
          if (pointSelNum < 10) {
            continue;
          }

          cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
          for (int i = 0; i < pointSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float s = 1;//20 * (pointOri.intensity - int(pointOri.intensity)); //1

            float srx = sin(s * transform[0]);
            float crx = cos(s * transform[0]);
            float sry = sin(s * transform[1]);
            float cry = cos(s * transform[1]);
            float srz = sin(s * transform[2]);
            float crz = cos(s * transform[2]);
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];

            float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z 
                      + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                      + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                      + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                      + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                      + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

            float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x 
                      + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z 
                      + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx) 
                      + s*tz*crx*cry) * coeff.x
                      + ((s*cry*crz - s*srx*sry*srz)*pointOri.x 
                      + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                      + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry) 
                      - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

            float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                      + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                      + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                      + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                      + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                      + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

            float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y 
                      - s*(crz*sry + cry*srx*srz) * coeff.z;
  
            float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y 
                      - s*(sry*srz - cry*crz*srx) * coeff.z;
  
            float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
          }
          cv::transpose(matA, matAt);
          matAtA = matAt * matA;
          matAtB = matAt * matB;
          cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

          if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {20, 20, 20, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {
              if (matE.at<float>(0, i) < eignThre[i]) {
                for (int j = 0; j < 6; j++) {
                  matV2.at<float>(i, j) = 0;
                }
                isDegenerate = true;
              } else {
                break;
              }
            }
            matP = matV.inv() * matV2;
          }

          if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
            ROS_INFO ("laser odometry degenerate");
          }

          transform[0] += matX.at<float>(0, 0);
          transform[1] += matX.at<float>(1, 0);
          transform[2] += matX.at<float>(2, 0);
          transform[3] += matX.at<float>(3, 0);
          transform[4] += matX.at<float>(4, 0);
          transform[5] += matX.at<float>(5, 0);

          ROS_WARN("check transform for one iteration");
          std::cout<<transform[0]<<std::endl;
          std::cout<<transform[1]<<std::endl;
          std::cout<<transform[2]<<std::endl;
          std::cout<<transform[3]<<std::endl;
          std::cout<<transform[4]<<std::endl;
          std::cout<<transform[5]<<std::endl;

          deltaR = sqrt(matX.at<float>(0, 0) * 180 / PI * matX.at<float>(0, 0) * 180 / PI
                       + matX.at<float>(1, 0) * 180 / PI * matX.at<float>(1, 0) * 180 / PI
                       + matX.at<float>(2, 0) * 180 / PI * matX.at<float>(2, 0) * 180 / PI);
          deltaT = sqrt(matX.at<float>(3, 0) * 100 * matX.at<float>(3, 0) * 100
                       + matX.at<float>(4, 0) * 100 * matX.at<float>(4, 0) * 100
                       + matX.at<float>(5, 0) * 100 * matX.at<float>(5, 0) * 100);

          if (deltaR < 0.05 && deltaT < 0.1) 
            break;

        }
          ROS_INFO("Front_end Node : iter: %d, deltaR: %f, deltaT: %f", iterCount, deltaR, deltaT);
        
      }
          
      float rx, ry, rz, tx, ty, tz;
      AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                         -transform[0], -transform[1], -transform[2], rx, ry, rz);// ? 1.05 //-transform[1] * 1.05    
      
      float x1 = cos(rz) * transform[3]// - imuShiftFromStartX) 
               - sin(rz) * transform[4];// - imuShiftFromStartY);
      float y1 = sin(rz) * transform[3]// - imuShiftFromStartX) 
               + cos(rz) * transform[4];// - imuShiftFromStartY);
      float z1 = transform[5];// * 1.05 - imuShiftFromStartZ;
      
      float x2 = x1;
      float y2 = cos(rx) * y1 - sin(rx) * z1;
      float z2 = sin(rx) * y1 + cos(rx) * z1;

      tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
      ty = transformSum[4] - y2;
      tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

      transformSum[0] = rx;
      transformSum[1] = ry;
      transformSum[2] = rz;
      transformSum[3] = tx;
      transformSum[4] = ty;
      transformSum[5] = tz;

      ROS_WARN("check transform once");
      std::cout<<transform[0]<<std::endl;
      std::cout<<transform[1]<<std::endl;
      std::cout<<transform[2]<<std::endl;
      std::cout<<transform[3]<<std::endl;
      std::cout<<transform[4]<<std::endl;
      std::cout<<transform[5]<<std::endl;

 //     ROS_WARN("check transformSum");
 //     std::cout<<transformSum[0]<<std::endl;
//      std::cout<<transformSum[1]<<std::endl;
//      std::cout<<transformSum[2]<<std::endl;
//      std::cout<<transformSum[3]<<std::endl;
 //     std::cout<<transformSum[4]<<std::endl;
//      std::cout<<transformSum[5]<<std::endl;

      geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[0], transformSum[1], transformSum[2]);

      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = geoQuat.x;
      laserOdometry.pose.pose.orientation.y = geoQuat.y;
      laserOdometry.pose.pose.orientation.z = geoQuat.z;
      laserOdometry.pose.pose.orientation.w = geoQuat.w;

      laserOdometry.pose.pose.position.x = transformSum[3];
      laserOdometry.pose.pose.position.y = transformSum[4];
      laserOdometry.pose.pose.position.z = transformSum[5];
      pubLaserOdometry.publish(laserOdometry); // laser_odom_to_init

/*
      Eigen::Quaternionf quat;
      quat.x() = laserOdometry.pose.pose.orientation.x;
      quat.y() = laserOdometry.pose.pose.orientation.y;
      quat.z() = laserOdometry.pose.pose.orientation.z;
      quat.w() = laserOdometry.pose.pose.orientation.w;
      Eigen::Matrix3f R_debug;
      R_debug = quat;
      Eigen::Vector3f rpy = R_to_rpy(R_debug);
*/
      int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
      for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
      }

      int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
      for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
      }

      frameCount++;
      if (frameCount >= skipFrameNum + 1) {
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }
      }

      pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();
      
        if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) 
        {
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }
     
      if (frameCount >= skipFrameNum + 1) {
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera_init";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera_init";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera_init";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);

        ros::Time time_end = ros::Time::now();
        std::cout<<"Front_end procedure time is : "<< time_end - time_start<<std::endl;
      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}