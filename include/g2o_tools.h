#pragma once
#include <g2o/core/base_vertex.h>
#include <g2o/core/sparse_optimizer.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>


class VertexSim3 : public g2o::BaseVertex<7, g2o::Vector7> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0, 0, 0, 0, 1.0;
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        g2o::Vector7::ConstMapType updateVec(update);
        _estimate += updateVec;
    }

    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}

};

class VertexSE3 : public g2o::BaseVertex<6, g2o::Vector6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0, 0, 0, 0;
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        g2o::Vector6::ConstMapType updateVec(update);
        _estimate += updateVec;
    }

    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}

};

/**
 * @brief Template skew-symmetric matrix Map of rotvec
 * 
 * @tparam T 
 * @param v 
 * @return g2o::MatrixN<3, T> 
 */
template <typename T>
g2o::MatrixN<3, T> skew(const g2o::VectorN<3, T> &v) {
    g2o::MatrixN<3, T> m;
    m.setZero();
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 2) = -v(0);
    m(1, 0) = v(2);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
}

/**
 * @brief Template Sim3 ExpMap
 * 
 * @tparam T 
 * @param update 
 * @return Rotation, Translation
 */
template <typename T>
std::tuple<g2o::MatrixN<3, T>, g2o::VectorN<3, T>, T> Sim3Exp(const T* update)
{
    g2o::VectorN<3, T> omega;
    g2o::VectorN<3, T> upsilon;
    for (int i = 0; i < 3; i++) omega[i] = update[i];
    for (int i = 0; i < 3; i++) upsilon[i] = update[i + 3];

    T theta = omega.norm();
    g2o::MatrixN<3, T> Omega = skew<T>(omega);

    g2o::MatrixN<3, T> R;
    g2o::MatrixN<3, T> V;
    g2o::MatrixN<3, T> I = g2o::MatrixN<3, T>::Identity();
    if (theta < T(1e-4))  // Use Taylor Series Approximation
    {
        g2o::MatrixN<3, T> Omega2 = Omega * Omega;
        R = (I + Omega + T(0.5) * Omega2);
        V = (I + T(0.5) * Omega + T(1.) / T(6.) * Omega2);
    }
    else 
    {
        g2o::MatrixN<3, T> Omega2 = Omega * Omega;
        T costh = cos(theta); // cos function Not Implemented By cmath !!! , see Jet.h
        T sinth = sin(theta); // sin function Not Implemented By cmath !!! , see Jet.h
        T invth2 = pow(theta, -2);  // pow function Not Implemented By cmath !!! , see Jet.h
        T invth3 = pow(theta, -3);  // pow function Not Implemented By cmath !!! , see Jet.h
        R = (I + sinth / theta * Omega +
            (T(1.) - costh) * invth2 * Omega2);
        V = (I +
            (T(1.) - costh) * invth2 * Omega +
            (theta - sinth) * invth3 * Omega2);
    }
    T s = update[6];
    return {R, V * upsilon, s};
}

/**
 * @brief Template SE3 ExpMap
 * 
 * @tparam T 
 * @param update 
 * @return std::tuple<g2o::MatrixN<3, T>, g2o::VectorN<3, T> > 
 */
template <typename T>
std::tuple<g2o::MatrixN<3, T>, g2o::VectorN<3, T> > SE3Exp(const T* update)
{
    g2o::VectorN<3, T> omega;
    g2o::VectorN<3, T> upsilon;
    for (int i = 0; i < 3; i++) omega[i] = update[i];
    for (int i = 0; i < 3; i++) upsilon[i] = update[i + 3];

    T theta = omega.norm();
    g2o::MatrixN<3, T> Omega = skew<T>(omega);

    g2o::MatrixN<3, T> R;
    g2o::MatrixN<3, T> V;
    g2o::MatrixN<3, T> I = g2o::MatrixN<3, T>::Identity();
    if (theta < T(1e-4))  // Use Taylor Series Approximation
    {
        g2o::MatrixN<3, T> Omega2 = Omega * Omega;
        R = (I + Omega + T(0.5) * Omega2);
        V = (I + T(0.5) * Omega + T(1.) / T(6.) * Omega2);
    }
    else 
    {
        g2o::MatrixN<3, T> Omega2 = Omega * Omega;
        T costh = cos(theta); // cos function Not Implemented By cmath !!! , see Jet.h
        T sinth = sin(theta); // sin function Not Implemented By cmath !!! , see Jet.h
        T invth2 = pow(theta, -2);  // pow function Not Implemented By cmath !!! , see Jet.h
        T invth3 = pow(theta, -3);  // pow function Not Implemented By cmath !!! , see Jet.h
        R = (I + sinth / theta * Omega +
            (T(1.) - costh) * invth2 * Omega2);
        V = (I +
            (T(1.) - costh) * invth2 * Omega +
            (theta - sinth) * invth3 * Omega2);
    }
    return {R, V * upsilon};
}

/**
 * @brief summary g2o edge erros: 
 * Min, Q25, Q50, Q75, Max, Mean
 * 
 * @tparam EdgeType 
 * @param optimizer g2o::SpaseOptimizer
 * @return std::map<std::string, double> 
 */
template <typename EdgeType>
std::map<std::string, double> g2oLogEdges(const g2o::SparseOptimizer &optimizer)
{
    std::map<std::string, double> info;
    std::vector<double> chiList;
    chiList.reserve(optimizer.edges().size());
    double chiMean = 0;
    for(g2o::HyperGraph::EdgeSet::const_iterator it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it)
    {
        EdgeType* e = dynamic_cast<EdgeType*>(*it);
        double err = sqrt(e->chi2());
        chiList.push_back(err);
        chiMean += err;
    }
    if(chiList.size() < 1)
        return info;
    if(chiList.size() < 2)
    {
        info.insert(std::make_pair("Edge",chiList[0]));
        return info;
    }
    int N = chiList.size();
    std::sort(chiList.begin(),chiList.end());
    info["Min"] = chiList[0];
    info["Q25"] = chiList[(int)(0.25*N)];
    info["Q50"] = chiList[(int)(0.5*N)];
    info["Q75"] = chiList[(int)(0.75*N)];
    info["Max"] = chiList[N-1];
    info["Mean"] = chiMean/N;
    return info;
}

/**
 * @brief summary g2o edge erros: 
 * Min, Q25, Q50, Q75, Max, Mean
 * 
 * @tparam EdgeType 
 * @param optimizer g2o::SparseOptimizer
 * @param condition_func condition function to filter edges to be computed
 * @return std::map<std::string, double> 
 */
template <typename EdgeType>
std::map<std::string, double> g2oLogEdges(const g2o::SparseOptimizer &optimizer, bool (* condition_func)(int edge_idx))
{
    std::map<std::string, double> info;
    std::vector<double> chiList;
    chiList.reserve(optimizer.edges().size());
    double chiMean = 0;
    for(g2o::HyperGraph::EdgeSet::const_iterator it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it)
    {
        if(!(condition_func((*it)->id())))
            continue;
        EdgeType* e = dynamic_cast<EdgeType*>(*it);
        double err = sqrt(e->chi2());
        chiList.push_back(err);
        chiMean += err;
    }
    if(chiList.size() < 1)
        return info;
    if(chiList.size() < 2)
    {
        info.insert(std::make_pair("Edge",chiList[0]));
        return info;
    }
    int N = chiList.size();
    std::sort(chiList.begin(),chiList.end());

    info["Min"] = chiList[0];
    info["Q25"] = chiList[(int)(0.25*N)];
    info["Q50"] = chiList[(int)(0.5*N)];
    info["Q75"] = chiList[(int)(0.75*N)];
    info["Max"] = chiList[N-1];
    info["Mean"] = chiMean/N;
    return info;
}

/**
 * @brief print a map class: 
 * Modified from https://en.cppreference.com/w/cpp/container/map
 * @param comment 
 * @param m 
 */
void print_map(std::string_view comment, const std::map<std::string, double>& m)
{
    std::cout << comment;
    // iterate using C++17 facilities
    for (const auto& [key, value] : m)
        std::cout << '[' << key << "] = " << value << "; ";
 
// C++11 alternative:
//  for (const auto& n : m)
//      std::cout << n.first << " = " << n.second << "; ";
//
// C++98 alternative
//  for (std::map<std::string, int>::const_iterator it = m.begin(); it != m.end(); it++)
//      std::cout << it->first << " = " << it->second << "; ";
 
    std::cout << '\n';
}