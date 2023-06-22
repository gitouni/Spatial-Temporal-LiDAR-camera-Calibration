#include "GPR.hpp"
#include "yaml-cpp/yaml.h"


/**
 * @brief load train_x, train_y, test_x, test_y
 * 
 * @param filename 
 * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::Vector2d, double> 
 */
std::tuple<std::vector<Eigen::Vector2d>, Eigen::VectorXd, Eigen::Vector2d, double, double, double> load_data(const std::string &filename)
{
    const YAML::Node node = YAML::LoadFile(filename);
    std::vector<std::vector<double>> train_x;
    auto train_x_node = node["train_x"];
    for(const auto &train_x_itemnode: train_x_node)
        train_x.push_back(train_x_itemnode.as<std::vector<double>>());
    std::vector<double> train_y = node["train_y"].as<std::vector<double>>();
    std::vector<double> test_x = node["test_x"].as<std::vector<double>>();
    double test_y = node["test_y"].as<double>();
    double sigma = node["sigma"].as<double>();
    double l = node["l"].as<double>();
    std::vector<Eigen::Vector2d> trX;
    Eigen::VectorXd trY;
    Eigen::Vector2d teX;
    trY.resize(train_y.size());
    for(const auto &trx : train_x)
    {
        Eigen::Vector2d trX_eig(trx.data());
        trX.push_back(std::move(trX_eig));
    }
    for(Eigen::Index ri = 0; ri < trY.rows(); ++ri)
        trY(ri) = train_y[ri];
    teX[0] = test_x[0];
    teX[1] = test_x[1];
    return {trX, trY, teX, test_y, sigma, l};
}


int main(int argc, char** argv)
{
    std::vector<Eigen::Vector2d> trX;
    Eigen::VectorXd trY;
    Eigen::Vector2d teX;
    double teY;
    double sigma, l;
    std::tie(trX, trY, teX, teY, sigma, l) = load_data("../debug/test_gpr.yml");
    std::cout << "trX:\n";
    for(auto const &trx:trX)
        std::cout << trx.transpose() << std::endl;
    std::cout << "trY:" << trY.transpose() << std::endl;
    std::cout << "teX:" << teX.transpose() << std::endl;
    std::cout << "teY:" << teY << std::endl;
    auto gpr_params = GPRParams();
    gpr_params.sigma = sigma;
    gpr_params.l = l;
    gpr_params.optimize = true;
    gpr_params.verborse = true;
    auto gpr = GPR(gpr_params);
    std::tie(sigma, l) = gpr.fit(trX, trY);
    std::printf("sigma: %lf, l: %lf\n", sigma, l);
    double preY = gpr.predict(teX);
    std::printf("predict y: %lf, real y: %lf ,knn y: %lf\n", preY, teY, trY[0]);

}