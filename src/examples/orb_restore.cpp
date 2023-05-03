#include "orb_slam/include/System.h"

int main(int argc, char **argv){
    if(argc != 5)
    {
        std::cout << "\033[31;1m Got " << argc-1 << " Parameters, expect 4.\033[0m" << std::endl;
        throw std::invalid_argument("Args: path_to_vocabulary path_to_settings path_to_keyframe_dir path_to_map_filename");
        exit(0);
    }
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,false);
    std::string KeyFrameDir(argv[3]), mapFile(argv[4]);
    SLAM.RestoreSystemFromFile(KeyFrameDir, mapFile);

}