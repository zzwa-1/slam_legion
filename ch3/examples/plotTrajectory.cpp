#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
#include <unistd.h>
#include <iostream>
// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// path to trajectory file 不知为何使用相对路径会报错
string trajectory_file = "/home/zzwa/sjtu/slam_zzwa/ch3/examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv)
{

  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;  //形成一个类似于矩阵数组
  /*
  其实上述的这段代码才是标准的定义容器方法，只是我们一般情况下定义容器的元素都是C++中的类型，所以可以省略，
  这是因为在C++11标准中，aligned_allocator管理C++中的各种数据类型的内存方法是一样的，可以不需要着重写出来。
  但是在Eigen管理内存和C++11中的方法是不一样的，所以需要单独强调元素的内存分配和管理。
  */
  ifstream fin(trajectory_file);
  cout<<!fin<<endl;
  if (!fin)  //正确读取则为true
  {
    cout << "cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }
 
  while (!fin.eof())  //到文件末尾为1，否则为0；
  {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;  //取出文件内容
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));  //平移向量
    poses.push_back(Twr);
  }
  //cout<<poses[0].matrix()<<endl;  //一定要有matrix这一项，不然欧式变换矩阵无法输出
  cout << "read total " << poses.size() << " pose entries" << endl;

  // draw trajectory in pangolin
  DrawTrajectory(poses);
  return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses)
{
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);  //创建的窗口名及尺寸
  //glEnable用于启用各种功能 openGL（open graphics library)
  glEnable(GL_DEPTH_TEST);  //启用深度测试，根据坐标的远近自动隐蔽被遮住的图形，启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
  glEnable(GL_BLEND);  //启用颜色混合
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  //和上一条语句搭配
  //构建观察相机对象，此处构建的相机为用于观测的相机，而非SLAM中的相机传感器
  /*
  相机参数projection_matrix对象构建如下
  OpenGlMatrixSpec ProjectionMatrix(int w, int h, 
  GLprecision fu, GLprecision fv, GLprecision u0, GLprecision v0, 
  GLprecision zNear, GLprecision zFar );
  w、h：相机的视野宽、高
  fu、fv、u0、v0相机的内参，对应《视觉SLAM十四讲》中内参矩阵的fx、fy、cx、cy
  zNear、zFar：相机的最近、最远视距
  */
  /*
  相机、视点初始坐标 modelview_matrix对象构建如下：
  OpenGlMatrix ModelViewLookAt(GLprecision x, GLprecision y, GLprecision z, 
  GLprecision lx, GLprecision ly, GLprecision lz, AxisDirection up);
  x、y、z：相机的初始坐标
  lx、ly、lz：相机视点的初始位置，也即相机光轴朝向
  up：相机自身那一轴朝上放置
  */
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),  //用于构建观察相机的内参系数
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));
  /*
  交互视图（View）用于显示相机观察到的信息内容，首先需要构建相机视图的句柄
  使用如下方式构建交互视图对象:
  SetBounds(Attach bottom, Attach top, Attach left, Attach right, double aspect)
  bottom、top：视图在视窗内的上下范围，依次为下、上，采用相对坐标表示（0：最下侧，1：最上侧）
  left、right：视图在视窗内的左右范围，依次为左、右，采用相对左边表示（0：最左侧，1：最右侧）
  aspect：视图的分辨率，也即分辨率:参数aspect取正值，将由前四个参数设置的视图大小来裁剪达到设置的分辨率，参数aspect取负值，将拉伸图像以充满由前四个参数设置的视图范围
  SetHandler（）：用于确定视图的相机句柄
  */
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) //视窗未被关闭为false
  {
    /*
    用于清空色彩缓冲区和深度缓冲区，刷新显示信息。若不使用清理，视窗将自动保留上一帧信息。
    */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //对交互视图对象进行激活相机
    d_cam.Activate(s_cam);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //设置这个 “  底色 ” 的，即所谓的背景颜色，glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
    glLineWidth(2);  //指定栅格化线条的宽度。初始值为1。
    /*
    size_t的全称应该是size type，就是说“一种用来记录大小的数据类型”。
    通常我们用sizeof(XXX)操作，这个操作所得到的结果就是size_t类型。
    */
    for (size_t i = 0; i < poses.size(); i++)  
    {
      // 画每个位姿的三个坐标轴
      Vector3d Ow = poses[i].translation();
      // Vector3d Ow = poses[i] * Vector3d(0, 0, 0);  //两种方式等价
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES); //在 glBegin(GL_LINES) 和 glEnd() 之间设置的点 , 会被自动当做线的两个端点
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < poses.size(); i++)
    {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000); // sleep 5 ms
  }
}
