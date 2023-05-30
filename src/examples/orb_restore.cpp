#include "orb_slam/include/System.h"
#include <fstream>
int main(int argc, char **argv){
    if(argc != 6)
    {
        std::cout << "\033[31;1m Got " << argc-1 << " Parameters, expect 5.\033[0m" << std::endl;
        throw std::invalid_argument("Args: path_to_vocabulary path_to_settings path_to_keyframe_dir path_to_map_filename debug_log");
        exit(0);
    }
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,false);
    std::string KeyFrameDir(argv[3]);
    const std::string mapFile(argv[4]), debugFile(argv[5]);
    SLAM.Shutdown();
    SLAM.RestoreSystemFromFile(KeyFrameDir, mapFile);
    std::vector<ORB_SLAM2::KeyFrame* > KeyFrames = SLAM.GetAllKeyFrames(true);
    std::sort(KeyFrames.begin(), KeyFrames.end(), ORB_SLAM2::KeyFrame::lId);
    auto KeyptMatched = KeyFrames[0]->GetMatchedKptIds(KeyFrames[1]);
    
    std::cout << "Matched pairs betwenn " << KeyFrames[0]->mnId << " & " << KeyFrames[1]->mnId << " : " << KeyptMatched.size() << std::endl;
    std::ofstream ofs(debugFile);
    for(auto it=KeyptMatched.begin(); it!=KeyptMatched.end(); ++it)
    {
        ofs << it->first << " " << it->second << std::endl;
    }
    ofs.close();

}