#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <ctime>

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType; //定义一个轨迹类型的数据类型
typedef Eigen::Matrix<double, 6, 1> Vector6d;//定义一个六维向量
clock_t start,last;

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 存储彩色图和深度图
    TrajectoryType poses;         // 相机位姿

    ifstream fin("./pose.txt");  //打开pose
    if (!fin) {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./%s/%d.%s"); //图像文件格式，这个文件格式与文件路径与文件命名有关
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取完整图片包括alpha通道,1读取彩色图像，0读取灰度图片

        double data[7] = {0};  //生成一个7维的零向量
        for (auto &d:data)
            fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2])); //注意开始的数据是反过来的。
        poses.push_back(pose);//存入5个位姿
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0; //现实世界中1米在深度图中存储为一个depthScale值
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);  //reserve()函数用来给vector预分配存储区大小，但不对该段内存进行初始化

    //遍历5张图片
    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];//分别将颜色图像和深度图像都提取出来
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 获取深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point; //三维向量
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p; //前三维度表示位置，后面表示颜色
                //opencv中图像的data数组表示把其颜色信息按行优先的方式展成的一维数组！
                //color.step等价于color.cols
                //color.channels()表示图像的通道数
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud); //绘制点云图像
    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }
    //生成一个pangolin窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);//使能遮挡渲染
    glEnable(GL_BLEND);//使用颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//使用(源图像的alpha值)作为权重因子，使用(1-源图像的alpha值)作为权重因子

    //设置相机状态
    //ProjectionMatrix()中参数依次为pangolin窗口的宽度、pangolin窗口的高度、fx、fy、cx、cy、最小深度值和最大深度值
    //ModelViewLookAt()中各参数依次为相机位置、被观察点位置和相机各轴正方向(000表示右下前)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    //设置相机窗口
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清除深度和颜色缓存
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
