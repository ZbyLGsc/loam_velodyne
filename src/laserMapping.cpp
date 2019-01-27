#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

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
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <iostream>

#include <Eigen/Eigen>
using namespace Eigen;
using namespace std;

Matrix3f ypr_to_R(const Vector3f& ypr)
{
    float c, s;
    Matrix3f Rz = Matrix3f::Zero();
    float y = ypr(0);
    c = cos(y);
    s = sin(y);
    Rz(0,0) =  c;
    Rz(1,0) =  s;
    Rz(0,1) = -s;
    Rz(1,1) =  c;
    Rz(2,2) =  1;

    Matrix3f Ry = Matrix3f::Zero();
    float p = ypr(1);
    c = cos(p);
    s = sin(p);
    Ry(0,0) =  c;
    Ry(2,0) = -s;
    Ry(0,2) =  s;
    Ry(2,2) =  c;
    Ry(1,1) =  1;

    Matrix3f Rx = Matrix3f::Zero();
    float r = ypr(2);
    c = cos(r);
    s = sin(r);
    Rx(1,1) =  c;
    Rx(2,1) =  s;
    Rx(1,2) = -s;
    Rx(2,2) =  c;
    Rx(0,0) =  1;

    Matrix3f R = Rz*Ry*Rx;
    return R;
}

Vector3f R_to_ypr(const Matrix3f& R)
{
    Vector3f n = R.col(0);
    Vector3f o = R.col(1);
    Vector3f a = R.col(2);

    Vector3f ypr(3);
    float y = atan2(n(1), n(0));
    float p = atan2(-n(2), n(0)*cos(y)+n(1)*sin(y));
    float r = atan2(a(0)*sin(y)-a(1)*cos(y), -o(0)*sin(y)+o(1)*cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr;
}

Matrix3d ypr_to_R(const Vector3d& ypr)
{
    double c, s;
    Matrix3d Rz = Matrix3d::Zero();
    double y = ypr(0);
    c = cos(y);
    s = sin(y);
    Rz(0,0) =  c;
    Rz(1,0) =  s;
    Rz(0,1) = -s;
    Rz(1,1) =  c;
    Rz(2,2) =  1;

    Matrix3d Ry = Matrix3d::Zero();
    double p = ypr(1);
    c = cos(p);
    s = sin(p);
    Ry(0,0) =  c;
    Ry(2,0) = -s;
    Ry(0,2) =  s;
    Ry(2,2) =  c;
    Ry(1,1) =  1;

    Matrix3d Rx = Matrix3d::Zero();
    double r = ypr(2);
    c = cos(r);
    s = sin(r);
    Rx(1,1) =  c;
    Rx(2,1) =  s;
    Rx(1,2) = -s;
    Rx(2,2) =  c;
    Rx(0,0) =  1;

    Matrix3d R = Rz*Ry*Rx;
    return R;
}

Vector3d R_to_ypr(const Matrix3d& R)
{
    Vector3d n = R.col(0);
    Vector3d o = R.col(1);
    Vector3d a = R.col(2);

    Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0)*cos(y)+n(1)*sin(y));
    double r = atan2(a(0)*sin(y)-a(1)*cos(y), -o(0)*sin(y)+o(1)*cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr;
}

const double PI = M_PI;//3.1415926;

const float scanPeriod = 0.05;//0.1

const int stackFrameNum = 1;
const int mapFrameNum = 1;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

bool newLaserCloudCornerLast = false;
bool newLaserCloudSurfLast = false;
bool newLaserCloudFullRes = false;
bool newLaserOdometry = false;

int laserCloudCenWidth = 10;//10;
int laserCloudCenHeight = 5;//5
int laserCloudCenDepth = 10;//10;
const int laserCloudWidth = 21;//21;//31;
const int laserCloudHeight =11;// 11;//11;
const int laserCloudDepth = 21;//21;//31;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerStack(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfStack(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerStack2(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfStack2(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSel(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCorr(new pcl::PointCloud<pcl::PointXYZI>());
//pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudProj(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr coeffSel(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurround(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurround2(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<pcl::PointXYZI>());

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCorner_toMap(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurf_toMap(new pcl::PointCloud<pcl::PointXYZI>());


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfArray[laserCloudNum];

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudAdd;
vector<int> remove_idx;

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerLiteArray[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfLiteArray[laserCloudNum];

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerArray2[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfArray2[laserCloudNum];

//pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>());//FLANN
//pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtreeCornerFromMap(new pcl::search::KdTree<pcl::PointXYZI> ());
pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtreeSurfFromMap(new pcl::search::KdTree<pcl::PointXYZI> ());

//pcl::search::KdTree<pcl::PointXYZ>


pcl::PointCloud<pcl::PointXYZI> laserCloudCornerStack_save;
pcl::PointCloud<pcl::PointXYZI> laserCloudSurfStack_save;


float transformSum[6] = {0};
float transformIncre[6] = {0};
float transformTobeMapped[6] = {0};
float transformBefMapped[6] = {0};
float transformAftMapped[6] = {0};

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};

float imuYaw[imuQueLength] = {0};
float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuRollLast = 0.0, imuPitchLast = 0.0, imuYawLast = 0.0;

const float pi_round = 3.10;
bool is_mapping_valid = true;
bool is_mapping_init = true;

void transformAssociateToMap()
{
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
           - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
           + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

  transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
  transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
  transformIncre[5] = z2;

/*
  ROS_WARN("check transformIncre ... ");
  cout<<transformIncre[3]<<endl;
  cout<<transformIncre[4]<<endl;
  cout<<transformIncre[5]<<endl;

  ROS_WARN("check transformSum ... ");
  cout<<transformSum[0]<<endl;
  cout<<transformSum[1]<<endl;
  cout<<transformSum[2]<<endl;
  cout<<transformSum[3]<<endl;
  cout<<transformSum[4]<<endl;
  cout<<transformSum[5]<<endl;

  ROS_WARN("check transformBefMapped ... ");
  cout<<transformBefMapped[0]<<endl;
  cout<<transformBefMapped[1]<<endl;
  cout<<transformBefMapped[2]<<endl;
  cout<<transformBefMapped[3]<<endl;
  cout<<transformBefMapped[4]<<endl;
  cout<<transformBefMapped[5]<<endl;

  ROS_WARN("check transformAftMapped ... ");
  cout<<transformAftMapped[0]<<endl;
  cout<<transformAftMapped[1]<<endl;
  cout<<transformAftMapped[2]<<endl;
  cout<<transformAftMapped[3]<<endl;
  cout<<transformAftMapped[4]<<endl;
  cout<<transformAftMapped[5]<<endl;
*/

  float sbcx = sin(transformSum[0]);
  float cbcx = cos(transformSum[0]);
  float sbcy = sin(transformSum[1]);
  float cbcy = cos(transformSum[1]);
  float sbcz = sin(transformSum[2]);
  float cbcz = cos(transformSum[2]);

  float sblx = sin(transformBefMapped[0]);
  float cblx = cos(transformBefMapped[0]);
  float sbly = sin(transformBefMapped[1]);
  float cbly = cos(transformBefMapped[1]);
  float sblz = sin(transformBefMapped[2]);
  float cblz = cos(transformBefMapped[2]);

  float salx = sin(transformAftMapped[0]);
  float calx = cos(transformAftMapped[0]);
  float saly = sin(transformAftMapped[1]);
  float caly = cos(transformAftMapped[1]);
  float salz = sin(transformAftMapped[2]);
  float calz = cos(transformAftMapped[2]);

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
            - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
            - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  
  float tmp_roll = -asin(srx);

  if(fabs(tmp_roll-transformTobeMapped[0])>=pi_round){
    if(tmp_roll<0)
      tmp_roll += 2 * M_PI;
    else
      tmp_roll -= 2 * M_PI;
  }      

  transformTobeMapped[0] = tmp_roll;

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
 
   float tmp_pitch = atan2(srycrx / cos(transformTobeMapped[0]), 
                                 crycrx / cos(transformTobeMapped[0]));
  
  if(fabs(tmp_pitch-transformTobeMapped[1])>=pi_round){
    if(tmp_pitch<0)
      tmp_pitch += M_PI;
    else
      tmp_pitch -= M_PI;
  }    
  transformTobeMapped[1] = tmp_pitch;

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
  
  float tmp_yaw = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                 crzcrx / cos(transformTobeMapped[0]));
    
  if(fabs(tmp_yaw-transformTobeMapped[2]) >= pi_round){
    if(tmp_yaw<0)
      tmp_yaw += M_PI;
    else
      tmp_yaw -= M_PI;
  }

  if( fabs(tmp_yaw - transformTobeMapped[2]) < 0.3) //fabs(imuYawLast - transformTobeMapped[2]) )
      transformTobeMapped[2] = tmp_yaw;
  else
    transformTobeMapped[2] = transformTobeMapped[2];
    
      //transformTobeMapped[2] = imuYawLast;    
//transformTobeMapped[2]  = imuYawLast;

  x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  transformTobeMapped[3] = transformAftMapped[3] 
                         - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] = transformAftMapped[5] 
                         - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
 
   /*                     
  float imuRollLast = 0.0, imuPitchLast = 0.0, imuYawLast = 0.0;
  
  if (imuPointerLast >= 0) {
    
    while (imuPointerFront != imuPointerLast) {
      if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
        break;
      }
      imuPointerFront = (imuPointerFront + 1) % imuQueLength;
    }

    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
      imuRollLast = imuRoll[imuPointerFront];
      imuPitchLast = imuPitch[imuPointerFront];
      imuYawLast = imuYaw[imuPointerFront];
    } else {
      int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
      float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
                       / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
                      / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

      imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
      imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
      imuYawLast = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
    }

  }
    
    transformTobeMapped[0] =  0.9 * transformTobeMapped[0] +  0.1 * imuRollLast;// + 0.1 * imuPitchLast;//0.998 imuRollLast;//0.9 *
    transformTobeMapped[1] =  0.9 * transformTobeMapped[1] +  0.1 * imuPitchLast;// + 0.1 * imuPitchLast;//0.998
    transformTobeMapped[2] =  0.9 * transformTobeMapped[2] +  0.1 * imuYawLast;// + 0.1 * imuRollLast;
*/
}

void transformUpdate()
{
  
  /*if (imuPointerLast >= 0) {
    //float imuRollLast = 0, imuPitchLast = 0;
    while (imuPointerFront != imuPointerLast) {
      if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
        break;
      }
      imuPointerFront = (imuPointerFront + 1) % imuQueLength;
    }

    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
      imuRollLast = imuRoll[imuPointerFront];
      imuPitchLast = imuPitch[imuPointerFront];
      imuYawLast = imuYaw[imuPointerFront];
    } else {
      int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
      float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
                       / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
                      / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

      //imuRollLast  =  imuRoll[imuPointerFront]   *  ratioFront  +  imuRoll[imuPointerBack]  * ratioBack;
      //imuPitchLast =  imuPitch[imuPointerFront]  *  ratioFront  +  imuPitch[imuPointerBack] * ratioBack;
      //imuYawLast   =  imuYaw[imuPointerFront]    *  ratioFront  +  imuYaw[imuPointerBack]   * ratioBack;
    }

    transformTobeMapped[0] =  transformTobeMapped[0];// +  0.1 * imuRollLast;// + 0.1 * imuPitchLast;//0.998 imuRollLast;//0.9 *
    transformTobeMapped[1] =  transformTobeMapped[1];// +  0.1 * imuPitchLast;// + 0.1 * imuPitchLast;//0.998
    transformTobeMapped[2] =  transformTobeMapped[2];// +  0.1 * imuYawLast;// + 0.1 * imuRollLast;
  } */

    //transformTobeMapped[0] =  transformTobeMapped[0];
    //transformTobeMapped[1] =  transformTobeMapped[1];
    //transformTobeMapped[2] =  transformTobeMapped[2];

  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i];
    transformAftMapped[i] = transformTobeMapped[i];
  }
}

void pointAssociateToMap(pcl::PointXYZI *pi, pcl::PointXYZI *po)
{
  float x1 = cos(transformTobeMapped[2]) * pi->x
           - sin(transformTobeMapped[2]) * pi->y;
  float y1 = sin(transformTobeMapped[2]) * pi->x
           + cos(transformTobeMapped[2]) * pi->y;
  float z1 = pi->z;

  float x2 = x1;
  float y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  float z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  po->x = cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2
        + transformTobeMapped[3];
  po->y = y2 + transformTobeMapped[4];
  po->z = -sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2
        + transformTobeMapped[5];
  
  po->intensity = pi->intensity;
}

void pointAssociateTobeMapped(pcl::PointXYZI *pi, pcl::PointXYZI *po)
{
  float x1 = cos(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) 
           - sin(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
  float y1 = pi->y - transformTobeMapped[4];
  float z1 = sin(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) 
           + cos(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);

  float x2 = x1;
  float y2 = cos(transformTobeMapped[0]) * y1 + sin(transformTobeMapped[0]) * z1;
  float z2 = -sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  po->x = cos(transformTobeMapped[2]) * x2
        + sin(transformTobeMapped[2]) * y2;
  po->y = -sin(transformTobeMapped[2]) * x2
        + cos(transformTobeMapped[2]) * y2;
  po->z = z2;
  po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudCornerLast2)
{
  timeLaserCloudCornerLast = laserCloudCornerLast2->header.stamp.toSec();

  laserCloudCornerLast->clear();
  pcl::fromROSMsg(*laserCloudCornerLast2, *laserCloudCornerLast);

  newLaserCloudCornerLast = true;
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudSurfLast2)
{
  timeLaserCloudSurfLast = laserCloudSurfLast2->header.stamp.toSec();

  laserCloudSurfLast->clear();
  pcl::fromROSMsg(*laserCloudSurfLast2, *laserCloudSurfLast);

  newLaserCloudSurfLast = true;
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);

  newLaserCloudFullRes = true;
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
{
  timeLaserOdometry = laserOdometry->header.stamp.toSec();

  double roll, pitch, yaw;
  geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
  //tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
  //tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
  tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  //transformSum[0] = -pitch;//-pitch;
  //transformSum[1] = -yaw;//-yaw;
  //transformSum[2] = roll;
  
  transformSum[0] = roll;
  transformSum[1] = pitch;
  transformSum[2] = yaw;

  transformSum[3] = laserOdometry->pose.pose.position.x;
  transformSum[4] = laserOdometry->pose.pose.position.y;
  transformSum[5] = laserOdometry->pose.pose.position.z;

  //transformSum[3] =  laserOdometry->pose.pose.position.x;
  //transformSum[4] =  laserOdometry->pose.pose.position.z;
  //transformSum[5] = -laserOdometry->pose.pose.position.y;


  newLaserOdometry = true;
}

bool imu_init = true;
Eigen::Matrix3f R_imu_init;
float roll = 0.0, pitch = 0.0, yaw = 0.0;

void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{

  Eigen::Quaternionf quater_imu;
  quater_imu.x() = imuIn->orientation.x;
  quater_imu.y() = imuIn->orientation.y;
  quater_imu.z() = imuIn->orientation.z;
  quater_imu.w() = imuIn->orientation.w;

  if(imu_init)
  {   
      imu_init = false;
      R_imu_init = quater_imu;
      return;
  }

  Eigen::Matrix3f R_imu;
  R_imu = quater_imu;
  Eigen::Matrix3f _R_imu = R_imu_init.inverse() * R_imu;

  Eigen::Vector3f ypr_imu = R_to_ypr(_R_imu);

//  cout<<"yaw: "<<ypr_imu(0)<<endl;
 // cout<<"pitch: "<<ypr_imu(1)<<endl;
  //cout<<"roll: "<<ypr_imu(2)<<endl;

  if(abs(yaw-ypr_imu(0))>=M_PI){
    if(ypr_imu(0)<0)
      ypr_imu(0) = 2* M_PI + ypr_imu(0);
    else
      ypr_imu(0) = -2* M_PI + ypr_imu(0);   
  }       

  if(abs(pitch-ypr_imu(1))>=M_PI){
    if(ypr_imu(1)<0)
      ypr_imu(1) = 2* M_PI + ypr_imu(1);
    else
      ypr_imu(1) = -2* M_PI + ypr_imu(1);   
  }      

  if(abs(roll-ypr_imu(2))>=M_PI){
    if(ypr_imu(2)<0)
      ypr_imu(2) = 2* M_PI + ypr_imu(2);
    else
      ypr_imu(2) = -2* M_PI + ypr_imu(2);   
  }      

  yaw = ypr_imu(0);
  pitch = ypr_imu(1);
  roll = ypr_imu(2);
  
  
  //imuYawLast = ypr_imu(0);
  //imuPitchLast = ypr_imu(1);
  //imuRollLast = ypr_imu(2);
  
  /*
  tf::Quaternion orientation;
  
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  */

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;
  
  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuYaw[imuPointerLast] =  yaw;//-roll;
  imuRoll[imuPointerLast] =  roll;//-roll;
  imuPitch[imuPointerLast] = pitch;// -pitch;
/*
  Eigen::Vector3f acc_w, gravity;
  acc_w   << imuIn->linear_acceleration.x,
             imuIn->linear_acceleration.y,
             imuIn->linear_acceleration.z;

  gravity << 0.0,
             0.0,
             9.80;

  acc_w = _R_imu * acc_w - gravity;

  float accX = acc_w(0);
  float accY = acc_w(1);
  float accZ = acc_w(2);

  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;
  */

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_corner_last", 1, laserCloudCornerLastHandler);

  ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>
                                          ("/laser_cloud_surf_last", 1, laserCloudSurfLastHandler);

  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry> 
                                     ("/laser_odom_to_init", 5, laserOdometryHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>
                                         ("/velodyne_cloud_3", 1, laserCloudFullResHandler);

  //ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/djiros/imu", 50, imuHandler);
  
  //ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);

  ros::Publisher pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2> 
                                         ("/laser_cloud_surround", 2);

  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2> 
                                        ("/velodyne_cloud_registered", 2);

  ros::Publisher pub_corner_map = nh.advertise<sensor_msgs::PointCloud2> ("/corner_in_map", 2);

  ros::Publisher pub_surf_map   = nh.advertise<sensor_msgs::PointCloud2> ("/surf_in_map", 2);

  //ros::Publisher pub1 = nh.advertise<sensor_msgs::PointCloud2> ("/pc3", 2);

  //ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2> ("/pc4", 2);

  ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 10);
  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/world";
  //odomAftMapped.child_frame_id = "/aft_mapped";

  //tf::TransformBroadcaster tfBroadcaster;
 // tf::StampedTransform aftMappedTrans;
  //aftMappedTrans.frame_id_ = "/camera_init";
 // aftMappedTrans.child_frame_id_ = "/aft_mapped";

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  pcl::PointXYZI pointOri, pointSel, pointProj, coeff;

  cv::Mat matA0(5, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matB0(5, 1, CV_32F, cv::Scalar::all(-1));
  cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));

  cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

  bool isDegenerate = false;
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterCorner;
  downSizeFilterCorner.setLeafSize(0.25, 0.25, 0.25);//Outdoor Testing 
  //downSizeFilterCorner.setLeafSize(0.3, 0.3, 0.3);//Outdoor Testing
  //downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);//Indoor

  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf;
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4); //Outdoor Testing
  //downSizeFilterSurf.setLeafSize(0.2, 0.2, 0.2); //Indoor

  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterMap;
  //downSizeFilterMap.setLeafSize(0.1, 0.1, 0.1);

  for (int i = 0; i < laserCloudNum; i++) {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudCornerArray2[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudSurfArray2[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }

  int frameCount = stackFrameNum - 1;
  int mapFrameCount = mapFrameNum - 1;
  ros::Rate rate(10);
  bool status = ros::ok();
  //bool mapping_success = true;

  while (status) {
    ros::spinOnce();
    /*&&
        fabs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.01 &&
        fabs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.01 &&
        fabs(timeLaserCloudFullRes - timeLaserOdometry) < 0.01) */
    if (newLaserCloudCornerLast && newLaserCloudSurfLast && newLaserCloudFullRes && newLaserOdometry ){
    //if(newLaserCloudCornerLast && newLaserCloudSurfLast && newLaserCloudFullRes)// && newLaserOdometry)
    //{
      newLaserCloudCornerLast = false;
      newLaserCloudSurfLast = false;
      newLaserCloudFullRes = false;
      newLaserOdometry = false;

      double time_start = ros::Time::now().toSec();
      
       /* 
        ROS_WARN("transformTobeMapped initial value check ");
        cout<<transformTobeMapped[0]<<endl;
        cout<<transformTobeMapped[1]<<endl;
        cout<<transformTobeMapped[2]<<endl;
        cout<<transformTobeMapped[3]<<endl;
        cout<<transformTobeMapped[4]<<endl;
        cout<<transformTobeMapped[5]<<endl;
        */
      frameCount++;
      if (frameCount >= stackFrameNum) {
        transformAssociateToMap();
        /*
        ROS_WARN("transformTobeMapped after associate to Map value check ");
        cout<<transformTobeMapped[0]<<endl;
        cout<<transformTobeMapped[1]<<endl;
        cout<<transformTobeMapped[2]<<endl;
        cout<<transformTobeMapped[3]<<endl;
        cout<<transformTobeMapped[4]<<endl;
        cout<<transformTobeMapped[5]<<endl;
        */
        int laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        for (int i = 0; i < laserCloudCornerLastNum; i++) {
          pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);
          //if(fabs(pointSel.z) > 0.5 )
              laserCloudCornerStack2->push_back(pointSel);
        }

        int laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        for (int i = 0; i < laserCloudSurfLastNum; i++) {
          pointAssociateToMap(&laserCloudSurfLast->points[i], &pointSel);
          //if(fabs(pointSel.z) > 1.0 )
              laserCloudSurfStack2->push_back(pointSel);
        }
      }

      //double t_bfe_if = ros::Time::now().toSec();
      //std::cout<<"FXXXXK HERE time? :"<< t_bfe_if - time_start<<std::endl;

      if (frameCount >= stackFrameNum) {
        frameCount = 0;

        pcl::PointXYZI pointOnYAxis;
        pointOnYAxis.x = 0.0;
        pointOnYAxis.y = 10.0;
        pointOnYAxis.z = 0.0;
        pointAssociateToMap(&pointOnYAxis, &pointOnYAxis);

        int centerCubeI = int((transformTobeMapped[3] + 25.0) / 50.0) + laserCloudCenWidth; // 25.0/50.0
        int centerCubeJ = int((transformTobeMapped[4] + 25.0) / 50.0) + laserCloudCenHeight;
        int centerCubeK = int((transformTobeMapped[5] + 25.0) / 50.0) + laserCloudCenDepth;

        if (transformTobeMapped[3] + 25.0 < 0) centerCubeI--;
        if (transformTobeMapped[4] + 25.0 < 0) centerCubeJ--;
        if (transformTobeMapped[5] + 25.0 < 0) centerCubeK--;

        int laserCloudValidNum = 0;
        int laserCloudSurroundNum = 0;
        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
          for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
            for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++) {
                laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                laserCloudValidNum++;
                laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                laserCloudSurroundNum++;
            }
          }
        }
        
        if(!is_mapping_valid)
            ROS_WARN("Mapping Node : Out of bound, discard ... ");
            
            for (int i = 0; i < laserCloudValidNum; i++) {
              //ROS_WARN("check ValidInd is: %d", laserCloudValidInd[i]);
              if(is_mapping_valid)
              {
                  *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];//no 2
                  *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
              }    
              
              laserCloudCornerArray[laserCloudValidInd[i]]->points.clear();
              laserCloudSurfArray[laserCloudValidInd[i]]->points.clear();
            }
          
        mapFrameCount++;
          
        if (mapFrameCount >= mapFrameNum) {
          mapFrameCount = 0;

          laserCloudSurround2->clear();
          *laserCloudSurround2 = *laserCloudCornerFromMap + *laserCloudSurfFromMap;
        
/*          Eigen::Vector3d ypr;
          double roll = M_PI / 2.0;
          ypr<< 0.0,
                0.0,
                roll;

          Eigen::Matrix3d R_ = ypr_to_R(ypr);*/
          Eigen::Matrix3d R_;
          R_ << 1,  0,  0,
                0, -1,  0,
                0,  0, -1;

          Eigen::Matrix4d RT = MatrixXd::Identity(4, 4);
          
          RT.block(0, 0, 3, 3) = R_;

          //ros::Time time_1 = ros::Time::now();
          pcl::transformPointCloud (*laserCloudSurround2, *laserCloudSurround, RT);
          /*ros::Time time_2 = ros::Time::now();
          ROS_WARN("[laser Mapping] time in transform map is %f", (time_2 - time_1).toSec());*/

          sensor_msgs::PointCloud2 laserCloudSurround_ros;
          pcl::toROSMsg(*laserCloudSurround, laserCloudSurround_ros);
          laserCloudSurround_ros.header.stamp = ros::Time().fromSec(timeLaserOdometry);
          laserCloudSurround_ros.header.frame_id = "/world";//"/camera_init";
          pubLaserCloudSurround.publish(laserCloudSurround_ros);
        }

        int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
        int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

        int laserCloudCornerStackNum2 = laserCloudCornerStack2->points.size();
        for (int i = 0; i < laserCloudCornerStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudCornerStack2->points[i], &laserCloudCornerStack2->points[i]);
        }

        int laserCloudSurfStackNum2 = laserCloudSurfStack2->points.size();
        for (int i = 0; i < laserCloudSurfStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudSurfStack2->points[i], &laserCloudSurfStack2->points[i]);
        }

        laserCloudCornerStack->clear();
      
        downSizeFilterCorner.setInputCloud(laserCloudCornerStack2);
        downSizeFilterCorner.filter(*laserCloudCornerStack);
        //*laserCloudCornerStack = *laserCloudCornerStack2;// For Indoor        
        int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfStack2);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
        int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

        
        double time1 = ros::Time::now().toSec();
        pcl::PointXYZI point_select_stack;
        pcl::PointCloud<pcl::PointXYZI> pc_tmp_;
        for (int i = 0; i < laserCloudCornerStackNum; i++) 
        {
              point_select_stack = laserCloudCornerStack->points[i];
              
              if(sqrt( (point_select_stack.x -transformTobeMapped[3]) * (point_select_stack.x - transformTobeMapped[3]) 
              + (point_select_stack.y -transformTobeMapped[4]) * (point_select_stack.y -transformTobeMapped[4]) ) > 50.0 ) //80.0
                  continue;
                  
               pc_tmp_.points.push_back(point_select_stack);
        }

        *laserCloudCornerStack = pc_tmp_;

        pc_tmp_.points.clear();
        for (int i = 0; i < laserCloudSurfStackNum; i++) 
        {
              point_select_stack = laserCloudSurfStack->points[i];
              
              if(sqrt( (point_select_stack.x -transformTobeMapped[3]) * (point_select_stack.x - transformTobeMapped[3]) 
              + (point_select_stack.y -transformTobeMapped[4]) * (point_select_stack.y -transformTobeMapped[4]) ) > 50.0 )
                  continue;
                  
               pc_tmp_.points.push_back(point_select_stack);
        }

        *laserCloudSurfStack = pc_tmp_;
        

        laserCloudCornerStackNum = laserCloudCornerStack->points.size();
        laserCloudSurfStackNum = laserCloudSurfStack->points.size();

        laserCloudCornerStack2->clear();
        laserCloudSurfStack2->clear();

        //ROS_WARN("check Map corner and surf point NUm");
        //cout<<"corner: "<<laserCloudCornerFromMapNum<<endl;
        //cout<<"surf: "<<laserCloudSurfFromMapNum<<endl;

        //ROS_WARN("check Stack corner and surf point NUm");
        //cout<<"corner: "<<laserCloudCornerStackNum2<<endl;
        //cout<<"surf: "<<laserCloudSurfStackNum2<<endl;

        //double t0_5 = ros::Time::now().toSec();
        //std::cout<<"down sample time is :"<< t0_5 - t0 << std::endl;
        //double t_debug = ros::Time::now().toSec();  
        float deltaR = 0.0, deltaT = 0.0;

        laserCloudCorner_toMap->clear();
        laserCloudSurf_toMap->clear();
        pcl::PointXYZI point_select;

//##################################################################################################################################

        for (int i = 0; i < laserCloudCornerFromMapNum; i++) 
        {
              point_select = laserCloudCornerFromMap->points[i];
              
              if(point_select.intensity < 0.95)
                  point_select.intensity = max(0.0, point_select.intensity - 0.025); // Indoor -0.05 
              //ROS_WARN("Check Point intensity: %f",point_select.intensity);
              if(point_select.intensity < 0.3 )
                  continue;
              //sqrt( (point_select.x -transformTobeMapped[3]) * (point_select.x - transformTobeMapped[3]) 
              //+ (point_select.y -transformTobeMapped[4]) * (point_select.y -transformTobeMapped[4]) ) > 50.0 ||  
               laserCloudCorner_toMap->points.push_back(point_select);

        }

        for (int i = 0; i < laserCloudSurfFromMapNum; i++) 
        {
              point_select = laserCloudSurfFromMap->points[i];

              if(point_select.intensity < 0.95)
                  point_select.intensity = max(0.0, point_select.intensity - 0.025); // Indoor -0.05
              //ROS_WARN("Check Point intensity: %f",point_select.intensity);
              if(point_select.intensity < 0.3)
                  continue;
              //sqrt( (point_select.x -transformTobeMapped[3]) * (point_select.x - transformTobeMapped[3]) 
              //+ (point_select.y -transformTobeMapped[4]) * (point_select.y -transformTobeMapped[4]) ) > 50.0 ||  
               laserCloudSurf_toMap->points.push_back(point_select);
                  
        }

        int laserCloudCorner_toMapNum = laserCloudCorner_toMap->points.size();
        int laserCloudSurf_toMapNum = laserCloudSurf_toMap->points.size();

        sensor_msgs::PointCloud2 corner_in_map;
        sensor_msgs::PointCloud2 surf_in_map;
        pcl::toROSMsg(*laserCloudCorner_toMap, corner_in_map);
        pcl::toROSMsg(*laserCloudSurf_toMap, surf_in_map);
        
        corner_in_map.header.frame_id = "/world";
        surf_in_map.header.frame_id = "/world";
        pub_corner_map.publish(corner_in_map);
        pub_surf_map.publish(surf_in_map);

        double time2 = ros::Time::now().toSec();
        //ROS_WARN("check added time %f", time2 - time1);
//##################################################################################################################################
/*
        double time_radius_1 = ros::Time::now().toSec();
        pcl::RadiusOutlierRemoval<pcl::PointXYZI> radius_filter;
        pcl::PointCloud<pcl::PointXYZI> radius_out;
        
        radius_filter.setInputCloud(laserCloudCorner_toMap);
        radius_filter.setRadiusSearch(0.2);
        radius_filter.setMinNeighborsInRadius (1);
        radius_filter.filter (radius_out);

        *laserCloudCorner_toMap = radius_out;

        radius_out.points.clear();
        radius_filter.setInputCloud(laserCloudSurf_toMap);
        radius_filter.setRadiusSearch(1.0);
        radius_filter.setMinNeighborsInRadius (3);
        radius_filter.filter (radius_out);

        *laserCloudSurf_toMap = radius_out;
        double time_radius_2 = ros::Time::now().toSec();
        ROS_WARN("radius_filter cost time is : %f", time_radius_2 - time_radius_1 );
*/

        *laserCloudCornerFromMap = *laserCloudCorner_toMap;
        *laserCloudSurfFromMap = *laserCloudSurf_toMap;

        ROS_WARN("laserCloudCorner_toMapNum: %d",laserCloudCorner_toMapNum);
        ROS_WARN("laserCloudSurf_toMapNum: %d",laserCloudSurf_toMapNum);

        if (laserCloudCorner_toMapNum > 10 && laserCloudSurf_toMapNum > 100) {
          //double t_set_input = ros::Time::now().toSec();
          kdtreeCornerFromMap->setInputCloud(laserCloudCorner_toMap);
          kdtreeSurfFromMap->setInputCloud(laserCloudSurf_toMap);
          //double t_set_input2 = ros::Time::now().toSec();
//          ROS_WARN("set KdTree Time is : %f", t_set_input2 - t_set_input);

//##################################################################################################
//##################################################################################################
          float cx, cy, cz;
          float a11, a12, a13, a22, a23, a33, ax, ay, az;
          int iterCount;
          for (iterCount = 0; iterCount < 15; iterCount++) //15
          {
            laserCloudOri->clear();
            coeffSel->clear();

            float x0, y0, z0, x1, y1, z1, x2, y2, z2;
            float a012, l12, la, lb, lc, ld2;
            float s;

            for (int i = 0; i < laserCloudCornerStackNum; i++) {
              pointOri = laserCloudCornerStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel);
              kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);//5
              
              if (pointSearchSqDis[4] < 2.5) {//2.5
                //ROS_WARN("Using corner information ... ");
                cx = 0;
                cy = 0; 
                cz = 0;
                
                for (int j = 0; j < 5; j++) {//5

                  cx += laserCloudCorner_toMap->points[pointSearchInd[j]].x;
                  cy += laserCloudCorner_toMap->points[pointSearchInd[j]].y;
                  cz += laserCloudCorner_toMap->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5; 
                cz /= 5;

                a11 = 0;
                a12 = 0; 
                a13 = 0;
                a22 = 0;
                a23 = 0; 
                a33 = 0;

                ax = 0;
                ay = 0;
                az = 0;

                for (int j = 0; j < 5; j++) {
                   ax = laserCloudCorner_toMap->points[pointSearchInd[j]].x - cx;
                   ay = laserCloudCorner_toMap->points[pointSearchInd[j]].y - cy;
                   az = laserCloudCorner_toMap->points[pointSearchInd[j]].z - cz;

                   a11 += ax * ax;
                   a12 += ax * ay;
                   a13 += ax * az;
                   a22 += ay * ay;
                   a23 += ay * az;
                   a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5; 
                a13 /= 5;
                a22 /= 5;
                a23 /= 5; 
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                  x0 = pointSel.x;
                  y0 = pointSel.y;
                  z0 = pointSel.z;
                  x1 = cx + 0.1 * matV1.at<float>(0, 0);
                  y1 = cy + 0.1 * matV1.at<float>(0, 1);
                  z1 = cz + 0.1 * matV1.at<float>(0, 2);
                  x2 = cx - 0.1 * matV1.at<float>(0, 0);
                  y2 = cy - 0.1 * matV1.at<float>(0, 1);
                  z2 = cz - 0.1 * matV1.at<float>(0, 2);

                  a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                             * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                             + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                             * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                             + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                             * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                  l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                  la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                           + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                  lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                           - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                           + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  ld2 = a012 / l12;

                  pointProj = pointSel;
                  pointProj.x -= la * ld2;
                  pointProj.y -= lb * ld2;
                  pointProj.z -= lc * ld2;

                  s = 1 - 0.9 * fabs(ld2);

                  coeff.x = s * la;
                  coeff.y = s * lb;
                  coeff.z = s * lc;
                  coeff.intensity = s * ld2;

                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }
            
            float pa, pb, pc, pd, ps;
            float pd2;
            //float s;

            bool planeValid = true;

            for (int i = 0; i < laserCloudSurfStackNum; i++) {
              pointOri = laserCloudSurfStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel); 
              kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); //5

              if (pointSearchSqDis[4] < 2.5) {  //2.5              
                for (int j = 0; j < 5; j++) {
                  matA0.at<float>(j, 0) = laserCloudSurf_toMap->points[pointSearchInd[j]].x;
                  matA0.at<float>(j, 1) = laserCloudSurf_toMap->points[pointSearchInd[j]].y;
                  matA0.at<float>(j, 2) = laserCloudSurf_toMap->points[pointSearchInd[j]].z;
                }
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                pa = matX0.at<float>(0, 0);
                pb = matX0.at<float>(1, 0);
                pc = matX0.at<float>(2, 0);
                pd = 1;
 
                ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                planeValid = true;
                
                for (int j = 0; j < 5; j++) {
                  if (fabs(pa * laserCloudSurf_toMap->points[pointSearchInd[j]].x +
                      pb * laserCloudSurf_toMap->points[pointSearchInd[j]].y +
                      pc * laserCloudSurf_toMap->points[pointSearchInd[j]].z + pd) > 0.2) {//0.2
                    planeValid = false;
                    break;
                  }
                }

                if (planeValid) {
                  pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                  pointProj = pointSel;
                  pointProj.x -= pa * pd2;
                  pointProj.y -= pb * pd2;
                  pointProj.z -= pc * pd2;

                  s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                          + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                  coeff.x = s * pa;
                  coeff.y = s * pb;
                  coeff.z = s * pc;
                  coeff.intensity = s * pd2;

                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }

          //t1_5 = ros::Time::now().toSec();
          //std::cout<<"pure search time is : " <<t1_5 - t1<<std::endl;
          
            float srx = sin(transformTobeMapped[0]);
            float crx = cos(transformTobeMapped[0]);
            float sry = sin(transformTobeMapped[1]);
            float cry = cos(transformTobeMapped[1]);
            float srz = sin(transformTobeMapped[2]);
            float crz = cos(transformTobeMapped[2]);

            int laserCloudSelNum = laserCloudOri->points.size();
            if (laserCloudSelNum < 50) {
              continue;
            }

            cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
            float arx, ary, arz;

            for (int i = 0; i < laserCloudSelNum; i++) {
              pointOri = laserCloudOri->points[i];
              coeff = coeffSel->points[i];

              arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                        + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                        + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

              ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                        + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                        + ((-cry*crz - srx*sry*srz)*pointOri.x 
                        + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

              arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                        + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                        + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

              matA.at<float>(i, 0) = arx;
              matA.at<float>(i, 1) = ary;
              matA.at<float>(i, 2) = arz;
              matA.at<float>(i, 3) = coeff.x;
              matA.at<float>(i, 4) = coeff.y;
              matA.at<float>(i, 5) = coeff.z;
              matB.at<float>(i, 0) = -coeff.intensity;
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
              //cout<<"check Eigen Values : \n"<<matE<<endl;
              isDegenerate = false;
              
              float eignThre[6] = {100, 100, 100, 50, 50, 50};//100 //50 both work good
              //float eignThre[6] = {100, 100, 100, 100, 100, 100};//100 //50 both work good
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
              //ROS_BREAK();
              ROS_INFO ("laser mapping degenerate");
            }
            

            /*
            if (fabs(matX.at<float>(0, 0)) < 0.5 &&
            fabs(matX.at<float>(1, 0)) < 0.5 &&
            fabs(matX.at<float>(2, 0)) < 0.5 &&
            fabs(matX.at<float>(3, 0)) < 1 &&
            fabs(matX.at<float>(4, 0)) < 1 &&
            fabs(matX.at<float>(5, 0)) < 1) {
            */
            transformTobeMapped[0] += matX.at<float>(0, 0);
            transformTobeMapped[1] += matX.at<float>(1, 0);
            transformTobeMapped[2] += matX.at<float>(2, 0);
            transformTobeMapped[3] += matX.at<float>(3, 0);
            transformTobeMapped[4] += matX.at<float>(4, 0);
            transformTobeMapped[5] += matX.at<float>(5, 0);
          //} else {
          //  ROS_INFO ("Mapping update out of bound");
          //}
            
//            ROS_WARN("transformTobeMapped after %d iteration ", iterCount);
           /* 
            cout<<transformTobeMapped[0]<<endl;
            cout<<transformTobeMapped[1]<<endl;
            cout<<transformTobeMapped[2]<<endl;
            cout<<transformTobeMapped[3]<<endl;
            cout<<transformTobeMapped[4]<<endl;
            cout<<transformTobeMapped[5]<<endl;
            */
            
            deltaR = sqrt(matX.at<float>(0, 0) * 180 / PI * matX.at<float>(0, 0) * 180 / PI
                         + matX.at<float>(1, 0) * 180 / PI * matX.at<float>(1, 0) * 180 / PI
                         + matX.at<float>(2, 0) * 180 / PI * matX.at<float>(2, 0) * 180 / PI);
            deltaT = sqrt(matX.at<float>(3, 0) * 100 * matX.at<float>(3, 0) * 100
                         + matX.at<float>(4, 0) * 100 * matX.at<float>(4, 0) * 100
                         + matX.at<float>(5, 0) * 100 * matX.at<float>(5, 0) * 100);

            ROS_INFO ("iter: %d, deltaR: %f, deltaT: %f", iterCount, deltaR, deltaT);
            if (deltaR < 0.1 && deltaT < 0.2)  //0.05 , 0.1 // 0.05,  0.05 
              break;
          }
          
          if(deltaR > 0.5 || deltaT > 0.5){
            is_mapping_valid = false;
            for (int i = 0; i < 6; i++) {
                  transformTobeMapped[i] = transformAftMapped[i];
              }
          }
          else
              is_mapping_valid = true;
              
          transformUpdate();
        }

          /*
          ROS_WARN("transform after mapping ...");
            cout<<transformTobeMapped[0]<<endl;
            cout<<transformTobeMapped[1]<<endl;
            cout<<transformTobeMapped[2]<<endl;
            cout<<transformTobeMapped[3]<<endl;
            cout<<transformTobeMapped[4]<<endl;
            cout<<transformTobeMapped[5]<<endl;
        */
        int cubeI, cubeJ, cubeK;
        int cubeInd;
        
        double time_dynamic_filter_1 = ros::Time::now().toSec();
        //ROS_WARN("debuf info 0 ... ");       
        for (int i = 0; i < laserCloudCornerStackNum; i++) {
          //ROS_WARN("debuf info 0.1 ... ");       
          pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
          //pointAssociateToMap(&laserCloudCornerStack_save.points[i], &pointSel);
//################################################################################################################
          pointSearchInd.clear();
          pointSearchSqDis.clear();
          pointSearchSqDis.push_back(10.0);
          //ROS_WARN("debuf info 0.2 ... ");       
          pointSel.intensity = 0.4;
          
          if (!is_mapping_init )
          {
             // ROS_WARN("debuf info 0.3 ... ");       
              kdtreeCornerFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); //5
              //ROS_WARN("debuf info 0.4 ... ");       
                //for(int i = 0; i < 5; i++)
                    if(pointSearchSqDis[0] < 0.2)//outdoor 0.5 //indoor 0.2
                        laserCloudCornerFromMap->points[pointSearchInd[0]].intensity = min(1.0,laserCloudCornerFromMap->points[pointSearchInd[0]].intensity + 0.1);
          }

          if(sqrt( (pointSel.x - transformTobeMapped[3]) * (pointSel.x - transformTobeMapped[3]) 
                 + (pointSel.y - transformTobeMapped[4]) * (pointSel.y - transformTobeMapped[4]) ) > 50.0 ) // 40.0
                  continue;
          
          cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

          if (pointSel.x + 25.0 < 0) cubeI--;
          if (pointSel.y + 25.0 < 0) cubeJ--;
          if (pointSel.z + 25.0 < 0) cubeK--;

          //if (cubeI >= 0 && cubeI < laserCloudWidth && 
          //    cubeJ >= 0 && cubeJ < laserCloudHeight && 
          //    cubeK >= 0 && cubeK < laserCloudDepth) {
            //ROS_WARN("check corner cubeInd %d", cubeInd);*/
          //}
          //else
            //cout<<"###############################################################################"<<endl;
          cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          
          if(pointSearchSqDis[0] > 0.15)//Indoor 0.04 , outdoor 0.06
               laserCloudCornerArray[cubeInd]->push_back(pointSel);
//################################################################################################################          
        }

        //ROS_WARN("debuf info 0.5 ... ");       
        for (int i = 0; i < laserCloudSurfStackNum; i++) {
          pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);
          pointSearchInd.clear();
          pointSearchSqDis.clear();
          pointSearchSqDis.push_back(10.0);
          //pointAssociateToMap(&laserCloudSurfStack_save.points[i], &pointSel);
          
          pointSel.intensity = 0.4;//0.5
          if (!is_mapping_init )
          { 
              kdtreeSurfFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); //5
               //for(int i = 0; i < 5; i++)
                  if(pointSearchSqDis[0] < 0.2)//Indoor 0.2 outdoor 0.3
                      laserCloudSurfFromMap->points[pointSearchInd[0]].intensity = min(1.0,laserCloudSurfFromMap->points[pointSearchInd[0]].intensity + 0.1);//0.15
          }
          
          if(sqrt( (pointSel.x - transformTobeMapped[3]) * (pointSel.x - transformTobeMapped[3]) 
                 + (pointSel.y - transformTobeMapped[4]) * (pointSel.y - transformTobeMapped[4]) ) > 50.0 ) //50
                  continue;

          cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

          if (pointSel.x + 25.0 < 0) cubeI--;
          if (pointSel.y + 25.0 < 0) cubeJ--;
          if (pointSel.z + 25.0 < 0) cubeK--;

          //if (cubeI >= 0 && cubeI < laserCloudWidth && 
          //    cubeJ >= 0 && cubeJ < laserCloudHeight && 
          //    cubeK >= 0 && cubeK < laserCloudDepth) {
            //ROS_WARN("check surf cubeInd %d", cubeInd);
          //}
          //else
          //  cout<<"###############################################################################"<<endl;
          cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          
          if(pointSearchSqDis[0] > 0.15)//Indoor 0.04 , Outdoor 0.15
                laserCloudSurfArray[cubeInd]->push_back(pointSel);

          }

        //double time_dynamic_filter_2 = ros::Time::now().toSec();
        //ROS_WARN("Dynamic filter time is %f", time_dynamic_filter_2 - time_dynamic_filter_1);

        if(is_mapping_init)
          is_mapping_init = false;
/*
        int ind;
        for (int i = 0; i < laserCloudValidNum; i++) {
          ind = laserCloudValidInd[i];

          laserCloudCornerArray2[ind]->clear();
          downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
          downSizeFilterCorner.filter(*laserCloudCornerArray2[ind]);

          laserCloudSurfArray2[ind]->clear();
          downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
          downSizeFilterSurf.filter(*laserCloudSurfArray2[ind]);

          pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = laserCloudCornerArray[ind];
          laserCloudCornerArray[ind] = laserCloudCornerArray2[ind];
          laserCloudCornerArray2[ind] = laserCloudTemp;

          laserCloudTemp = laserCloudSurfArray[ind];
          laserCloudSurfArray[ind] = laserCloudSurfArray2[ind];
          laserCloudSurfArray2[ind] = laserCloudTemp;
        }
*/
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudFullRes3.header.frame_id = "/world";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
        
        /*
        ROS_WARN("check pub odom");
            cout<<transformAftMapped[0]<<endl;
            cout<<transformAftMapped[1]<<endl;
            cout<<transformAftMapped[2]<<endl;
            cout<<transformAftMapped[3]<<endl;
            cout<<transformAftMapped[4]<<endl;
            cout<<transformAftMapped[5]<<endl;
        */
        //geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
        //                          (transformAftMapped[0], transformAftMapped[1], transformAftMapped[2]);
        Eigen::Vector3f ypr_aftmap;
        ypr_aftmap << transformAftMapped[2],
                      transformAftMapped[1],
                      transformAftMapped[0];
        Eigen::Matrix3f R_odom = ypr_to_R(ypr_aftmap);
        Eigen::Quaternionf q_odom;
        q_odom = R_odom;
        
        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);//(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = q_odom.x(); //geoQuat.x; //-geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = q_odom.y(); //geoQuat.y; //-geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = q_odom.z(); //geoQuat.z; //geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = q_odom.w(); //geoQuat.w; //geoQuat.w;

        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        
        //if(is_mapping_valid)
            pubOdomAftMapped.publish(odomAftMapped);

        double time_end = ros::Time::now().toSec();

        //std::cout<<"final part time is : "<< time_end - t4<<std::endl;
        std::cout<<"Mapping procedure time is : "<< time_end - time_start<<std::endl;
      }
    }

    status = ros::ok();
    rate.sleep();

  }


  return 0;
}

