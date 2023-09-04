#include "backend_opt.h"
#include "io_tools.h"
#include "kitti_tools.h"
#include <yaml-cpp/yaml.h>
#include "argparse.hpp"


int main(int argc, char** argv){
    argparse::ArgumentParser parser("Back-end optimization of F-LOAM");
    parser.add_argument("--config").help("config file").required();
    parser.add_argument("--raw_pose").help("raw pose of the F-LOAM").required();
    parser.add_argument("--velo_dir").help("directory of velodyne pcd files").required();
    parser.parse_args(argc, argv);
    std::string yaml_file(parser.get<std::string>("--config"));
    assert(file_exist(yaml_file));
    std::string input_pose_file(parser.get<std::string>("--raw_pose"));
    std::string input_lidar_dir(parser.get<std::string>("--velo_dir"));
   
    auto option = BackEndOption();
    YAML::Node config = YAML::LoadFile(yaml_file);
    YAML::Node gui_config = config["gui"];
    YAML::Node io_config = config["io"];
    YAML::Node odom_config = config["odom"];
    YAML::Node loop_config = config["loopclosure"];
    YAML::Node multiway_config = config["multiway"];
    std::string output_pose_graph = io_config["mr_graph"].as<std::string>();
    std::string output_isam_poses = io_config["isam_poses"].as<std::string>();
    std::string output_mr_poses = io_config["mr_poses"].as<std::string>();
    bool save_map = io_config["save_map"].as<bool>();
    std::string map_path = io_config["map_path"].as<std::string>();

    assert(output_pose_graph.substr(output_pose_graph.find_last_of('.') + 1) == "json"); // suffix of a Posegraph must be '.json', or your effort is in vain!
    
    option.verborse = gui_config["verborse"].as<bool>();
    option.vis = gui_config["vis"].as<bool>();

    option.voxel = odom_config["voxel"].as<double>();
    option.keyframeMeterGap = odom_config["keyframeMeterGap"].as<double>();
    option.keyframeRadGap = odom_config["keyframeRadGap"].as<double>();
    option.icp_corase_dist = odom_config["icp_corase_dist"].as<double>();
    option.icp_refine_dist = odom_config["icp_refine_dist"].as<double>();
    option.icp_max_iter = odom_config["icp_max_iter"].as<int>();

    option.LCkeyframeMeterGap = loop_config["LCkeyframeMeterGap"].as<double>();
    option.LCkeyframeRadGap = loop_config["LCkeyframeRadGap"].as<double>();
    option.LCSubmapSize = loop_config["LCSubmapSize"].as<int>();
    option.loopOverlapThre = loop_config["loopOverlapThre"].as<double>();
    option.loopInlierRMSEThre = loop_config["loopInlierRMSEThre"].as<double>();

    option.MRmaxIter = multiway_config["MRmaxIter"].as<int>();
    option.MRmaxCorrDist = multiway_config["MRmaxCorrDist"].as<double>();
    option.MREdgePruneThre = multiway_config["MREdgePruneThre"].as<double>();
    bool odom_refine = multiway_config["RefineOdom"].as<bool>();
    bool use_multiway = multiway_config["use"].as<bool>();
    std::cout << "Parameters have been loaded from " << yaml_file << "." << std::endl;
    BackEndOptimizer optimizer(option);
    std::thread p(&BackEndOptimizer::LoopClosureRegThread, &optimizer);
    optimizer.LoopClosureRun(input_pose_file, input_lidar_dir);
    optimizer.StopLoopClosure();
    p.join();
    optimizer.UpdateISAM();
    optimizer.writePoses(output_isam_poses);
    std::cout << "isam optimized poses saved to " << output_isam_poses << std::endl;
    // *** Results of ISAM and Multway are too similar! ***
    if(use_multiway)
    {
        optimizer.MultiRegistration(odom_refine);
        optimizer.writePoseGraph(output_pose_graph);
        optimizer.UpdatePosesFromPG();
        optimizer.writePoses(output_mr_poses);        
    }
    if(save_map)
        optimizer.SaveMap(map_path);
}