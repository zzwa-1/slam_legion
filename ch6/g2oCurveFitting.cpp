#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace g2o;

#include <Eigen/Core>  //Eigen核心模块

using namespace Eigen;

#include <opencv2/core/core.hpp>

using namespace cv;


//曲线模型的顶点
//模板参数：优化变量维度和数据类型
class CurveFittingVertex : public BaseVertex<3, Eigen::Vector3d>
//:表示继承，public表示公有继承；CurveFittingVertex是派生类，BaseVertex<3, Eigen::Vector3d>是基类
{
public:  //以下定义的成员变量和成员函数都是公有的！
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  //解决Eigen库数据结构内存对齐问题

    //重置
    virtual void setToOriginImpl() override  //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
    {
        _estimate << 0, 0, 0;
    }

    //更新
    virtual void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    //存盘和读盘：留空
    virtual bool read(istream &in) {}  //istream类是c++标准输入流的一个基类
    virtual bool write(ostream &out) const {}  //ostream类是c++标准输出流的一个基类
};


//曲线模型的边
//模板参数：观测值维度、类型、连接顶点类型
class CurveFittingEdge : public BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  //解决Eigen库数据结构内存对齐问题

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}  //使用列表赋初值

    //计算曲线模型误差
    virtual void computeError() override  //virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    //计算雅可比矩阵
    virtual void linearizeOplus() override  //virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream& in) {}
    virtual bool write(ostream& out) const {}

public:
    double _x;  //x值，y值为_measurement

};


int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;  //真实参数值！
    double ae = 2.0, be = -1.0, ce = 5.0;  //估计参数值，并赋初值！

    //生成测量数据
    int N = 100;  //数据点数目
    double w_sigma = 1.0;  //测量噪声标准差
    double inv_sigma = 1.0 / w_sigma;  //测量噪声标准差的逆
    cv::RNG rng;  //OpenCV随机数生成器
    vector<double> x_data, y_data;
    for(int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }


    //构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  //起别名，BlockSolverTraits<3, 1>表示误差项维数
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  //线性求解器类型

    //梯度下降方法，可以从GN、LM、DogLeg中选
    auto solver = new OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    //c++中的make_unique表示智能指针类型，而g2o中的make_unique表示？？？
    g2o::SparseOptimizer optimizer;  //图模型
    optimizer.setAlgorithm(solver);  //设置求解器
    optimizer.setVerbose(true);  //向屏幕上显示优化过程信息

    //往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));  //ae、be和ce表示优化变量
    v->setId(0);
    optimizer.addVertex(v);

    //往图中增加边
    for(int i = 0; i < N; i++)
    {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);  //设置连接的顶点
        edge->setMeasurement(y_data[i]);  //设置测量值
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));  //信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
    }

    //执行优化
    cout << "开始优化！" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();  //优化过程初始化
    optimizer.optimize(10);  //设置优化的迭代次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "执行优化花费的时间为：" << time_used.count() << "秒！" << endl;

    //输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "优化变量ae、be和ce的最佳结果为：" << abc_estimate[0] << ", " << abc_estimate[1] << ", " << abc_estimate[2] << "!" << endl;

    return 0;
}

