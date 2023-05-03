#include <Eigen/Dense>
#include <vector>
#include <iostream>
/**
 * @brief Hand-eye calibration (from B to A)
 * 
 * @param TaList Motions of Sensor A
 * @param TbList Motions of Sensor B
 * @return Eigen::Matrix4d TAB (transformation from B to A)
 */
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> HECalib(std::vector<Eigen::Isometry3d> &TaList, std::vector<Eigen::Isometry3d> &TbList){
    assert(TaList.size()==TbList.size());
    std::vector<Eigen::Vector3d> alphaList, betaList;
    Eigen::MatrixX4d A(3*TaList.size(), 4);  // 3N, 4
    Eigen::VectorXd b(3*TaList.size());  // 3N
    Eigen::Vector3d alpha_mean, beta_mean;
    alpha_mean.setZero();
    beta_mean.setZero();
    for(std::size_t i = 0; i < TaList.size(); ++i){
        // Ta
        Eigen::AngleAxisd ax;
        ax.fromRotationMatrix(TaList[i].rotation());
        Eigen::Vector3d alpha = ax.angle() * ax.axis();
        alphaList.push_back(alpha);
        // Tb
        alpha_mean += alpha;
        ax.fromRotationMatrix(TbList[i].rotation());
        Eigen::Vector3d beta = ax.angle() * ax.axis();
        betaList.push_back(beta);
        beta_mean += beta;
    }
    alpha_mean /= TaList.size();
    beta_mean /= TbList.size();
    // Decentralization and Compute Covariance Matrix
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for(std::vector<Eigen::Vector3d>::const_iterator ita=alphaList.begin(), itb=betaList.begin(); ita!=alphaList.end() ;++ita, ++itb){
        H += (*itb - beta_mean) * (*ita - alpha_mean).transpose(); // (3,1) x (1,3) -> (3,3)
    }
    // Computate Rotation Part RAB
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d Ut = svd.matrixU().transpose(), Vt = svd.matrixV().transpose();
    Eigen::Matrix3d RAB = Vt.transpose() * Ut;
    if(RAB.determinant() < 0){
        Vt.row(2) *= -1;
        RAB = Vt.transpose() * Ut;
    }
    // Compute Translation Part tAB
    for(std::size_t i = 0; i < TaList.size(); ++i){
        A.block<3, 3>(3*i, 0) = TaList[i].rotation() - Eigen::Matrix3d::Identity();
        A.block<3, 1>(3*i, 3) = TaList[i].translation();
        b.block<3, 1>(3*i, 0) = RAB * TbList[i].translation();
    }
    Eigen::Vector4d res = (A.transpose() * A).ldlt().solve(A.transpose() * b);  // use cholesky decompostion to solve this problem with large
    return std::make_tuple(RAB, res.block<3, 1>(0, 0), res(3));

}