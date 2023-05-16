#include "backend_opt.h"
#include "io_tools.h"
#include "kitti_tools.h"
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "Expected args: yaml pose_txt Lidar_dir" << std::endl;
    }
    std::string yaml_file(argv[1]);
    assert(file_exist(yaml_file));
    std::string input_pose_file(argv[2]);
    std::string input_lidar_dir(argv[3]);
   
    auto option = BackEndOption();
    YAML::Node config = YAML::LoadFile(yaml_file);

    std::string output_pose_graph = config["io"]["mr_graph"].as<std::string>();
    std::string output_isam_poses = config["io"]["isam_poses"].as<std::string>();
    std::string output_mr_poses = config["io"]["mr_poses"].as<std::string>();
    assert(output_pose_graph.substr(output_pose_graph.find_last_of('.') + 1) == "json"); // suffix of a Posegraph must be '.json', or your effort is in vain!
    
    option.verborse = config["gui"]["verborse"].as<bool>();
    option.vis = config["gui"]["vis"].as<bool>();

    option.voxel = config["odom"]["voxel"].as<double>();
    option.keyframeMeterGap = config["odom"]["keyframeMeterGap"].as<double>();
    option.keyframeRadGap = config["odom"]["keyframeRadGap"].as<double>();
    option.icp_corase_dist = config["odom"]["icp_corase_dist"].as<double>();
    option.icp_refine_dist = config["odom"]["icp_refine_dist"].as<double>();
    option.icp_max_iter = config["odom"]["icp_max_iter"].as<int>();

    option.LCkeyframeMeterGap = config["loopclosure"]["LCkeyframeMeterGap"].as<double>();
    option.LCkeyframeRadGap = config["loopclosure"]["LCkeyframeRadGap"].as<double>();
    option.LCSubmapSize = config["loopclosure"]["LCSubmapSize"].as<int>();
    option.loopOverlapThre = config["loopclosure"]["loopOverlapThre"].as<double>();
    option.loopInlierRMSEThre = config["loopclosure"]["loopInlierRMSEThre"].as<double>();

    option.MRmaxIter = config["multiway"]["MRmaxIter"].as<int>();
    option.MRmaxCorrDist = config["multiway"]["MRmaxCorrDist"].as<double>();
    option.MREdgePruneThre = config["multiway"]["MREdgePruneThre"].as<double>();
    bool odom_refine = config["multiway"]["RefineOdom"].as<bool>();
    bool use_multiway = config["multiway"]["use"].as<bool>();
    std::cout << "Parameters have been loaded from " << yaml_file << "." << std::endl;
    BackEndOptimizer optimizer(option);
    std::thread p(&BackEndOptimizer::LoopClosureRegThread, &optimizer);
    optimizer.LoopClosureRun(input_pose_file, input_lidar_dir);
    optimizer.StopLoopClosure();
    p.join();
    optimizer.UpdateISAM();
    optimizer.writePoses(output_isam_poses);
    // *** Results of ISAM and Multway are too similar! ***
    if(use_multiway){
        optimizer.MultiRegistration(odom_refine);
        optimizer.writePoseGraph(output_pose_graph);
        optimizer.UpdatePosesFromPG();
        optimizer.writePoses(output_mr_poses);        
    }
}