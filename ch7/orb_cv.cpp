#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

string path1 = "/home/zzwa/sjtu/slam_legion/ch7/1.png";  //图片1路径
string path2 = "/home/zzwa/sjtu/slam_legion/ch7/2.png";  //图片2路径

int main(int argc, char **argv) {

  //-- 读取图像
  Mat img_1 = imread(path1, CV_LOAD_IMAGE_COLOR);  //表示返回一张彩色图
  Mat img_2 = imread(path2, CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr); //assert()为断言函数，如果它的条件返回错误，则终止程序执行

  //-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;  //提取两张图片的关键点
  Mat descriptors_1, descriptors_2;  //描述子
  Ptr<FeatureDetector> detector = ORB::create();  //可以修改特征点的个数来增加匹配点的数量，创建特征检测器
  Ptr<DescriptorExtractor> descriptor = ORB::create();  //创建描述子提取器
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");  //基于汉明距离实现特征匹配

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2); //检测两张图片里的fast角点

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);  //根据两张图片的角点位置计算描述子
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1;  //定义ORB特征显示结果的变量
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);  //画出图像1的ORB特征点提取结果
  imshow("ORB features", outimg1);  //显示标记完特征点的图片

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);  //描述子1和2进行匹配
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  //minmax_element()为c++中定义的寻找最小值和最大值的函数。
  //第3个参数表示比较函数，默认从小到大，可以省略
  double min_dist = min_max.first->distance;  //将两幅图像的ORB特征点之间的最小距离赋值给min_dist
  double max_dist = min_max.second->distance;  //将两幅图像的ORB特征点之间的最大距离赋值给max_dist

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;  //用来存储好的匹配
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}
