#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

ros::Publisher pubVinsOdom;
ros::Publisher pubVinsPath;

nav_msgs::Path vinsPath_tran;

Matrix3d _R;
Vector3d _T;

pair<Matrix3d, Vector3d> getRT()
{
    Matrix3d R;
    Vector3d T;

    Vector3d    Tv, Tl;
    Quaterniond qv, ql;

    qv.w() = 0.999692868318;
    qv.x() = 0.0224451207307; 
    qv.y() = 0.00332297762905; 
    qv.z() = 0.00996711643554;

    Tv(0) =  9.11141848414;
    Tv(1) =  5.90805467186;
    Tv(2) =  1.47811580872;

    ql.w() =  0.99974686086;
    ql.x() = -0.0123135596216;
    ql.y() = -0.0161103368711;
    ql.z() = 0.00974923051868;
    
    Tl(0) =  9.42306144022;
    Tl(1) =  5.89581846139;
    Tl(2) =  0.844748345697;

    Matrix3d Rv;
    Rv = qv;
    Matrix3d Rl;
    Rl = ql;

    R = Rl * Rv.inverse();
    T = Tl - R * Tv;

    return make_pair(R, T);
}

void VinsOdomCallBack(nav_msgs::Odometry vinsOdom_cur)
{

    Vector3d T_vins_cur, T_vins_tran;
    Quaterniond Q_vins_cur;

    T_vins_cur(0) = vinsOdom_cur.pose.pose.position.x;
    T_vins_cur(1) = vinsOdom_cur.pose.pose.position.y;
    T_vins_cur(2) = vinsOdom_cur.pose.pose.position.z;
    
    Q_vins_cur.w() = vinsOdom_cur.pose.pose.orientation.w;
    Q_vins_cur.x() = vinsOdom_cur.pose.pose.orientation.x;
    Q_vins_cur.y() = vinsOdom_cur.pose.pose.orientation.y;
    Q_vins_cur.z() = vinsOdom_cur.pose.pose.orientation.z;

    Matrix3d R_vins_cur;
    R_vins_cur  = Q_vins_cur;

    Matrix3d R_vins_tran = _R * R_vins_cur;
    Quaterniond Q_vins_tran;
    Q_vins_tran = R_vins_tran;

    T_vins_tran = _R * T_vins_cur + _T;

    nav_msgs::Odometry vinsOdom_tran;

    vinsOdom_tran.header = vinsOdom_cur.header;
    vinsOdom_tran.pose.pose.position.x = T_vins_tran(0);
    vinsOdom_tran.pose.pose.position.y = T_vins_tran(1);
    vinsOdom_tran.pose.pose.position.z = T_vins_tran(2);
    
    vinsOdom_tran.pose.pose.orientation.w = Q_vins_tran.w();
    vinsOdom_tran.pose.pose.orientation.x = Q_vins_tran.x();
    vinsOdom_tran.pose.pose.orientation.y = Q_vins_tran.y();
    vinsOdom_tran.pose.pose.orientation.z = Q_vins_tran.z();
    pubVinsOdom.publish(vinsOdom_tran);

    vinsPath_tran.header = vinsOdom_cur.header;
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = T_vins_tran(0);
    pose.pose.position.y = T_vins_tran(1);
    pose.pose.position.z = T_vins_tran(2);
    pose.pose.orientation.w = Q_vins_tran.w();
    pose.pose.orientation.x = Q_vins_tran.x();
    pose.pose.orientation.y = Q_vins_tran.y();
    pose.pose.orientation.z = Q_vins_tran.z();
    pose.header = vinsPath_tran.header;
    vinsPath_tran.poses.push_back(pose);
    pubVinsPath.publish(vinsPath_tran);        

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laser_vins_compare");
  ros::NodeHandle nh;

  pair<Matrix3d, Vector3d> R_T = getRT();
  _R = R_T.first;
  _T = R_T.second;

  ros::Subscriber subVinsOdom = nh.subscribe<nav_msgs::Odometry>
                                            ("/self_calibration_estimator/odometry", 10, VinsOdomCallBack);

  pubVinsOdom = nh.advertise<nav_msgs::Odometry> ("/VinsOdom", 10);

  pubVinsPath = nh.advertise<nav_msgs::Path> ("/VinsPath", 10);

  ros::spin();
}