/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <kitti_tools.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "orb_slam/include/System.h"

using namespace std;


void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        std::cout << "\033[31;1m Got " << argc-1 << " Parameters, expect 3.\033[0m" << std::endl;
        throw std::invalid_argument("Args: path_to_vocabulary path_to_settings path_to_sequence");
        exit(0);
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string sequenceDir = argv[3];
    checkpath(sequenceDir);
    LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);
    std::cout << "Find " << vstrImageFilenames.size() << " images." << endl;
    std::size_t nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,false);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << endl << "-------" << endl;
    std::cout << "Start processing sequence ..." << endl;
    std::cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    // bool bLocalMapBusy = false, bLoopClosingBusy = false;
    // >>>>>>>>>>>>>>>> Replace 10 with nImages <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    for(std::size_t ni=0; ni < nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }


        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        // Wait until the underlying threads finish tasks.
        // std::tie(bLocalMapBusy, bLoopClosingBusy) = SLAM.isRequiredWait();
        // if(bLocalMapBusy || bLoopClosingBusy){
        //     if(bLocalMapBusy){
        //         if(!bLoopClosingBusy)
        //             std::cout << "Wait for Local Map..." << std::endl;
        //         else
        //             std::cout << "Wait Local Mapping and Loop Closing..." << std::endl;
        //         }else{
        //             std::cout << "Wait for Loop Closing..." << std::endl;
        //         }
        //         while(1){
        //             std::tie(bLocalMapBusy, bLoopClosingBusy) = SLAM.isRequiredWait();
        //             if(!(bLocalMapBusy || bLoopClosingBusy)){
        //                 break;
        //             }
        //             usleep(5000);
        //         }
        // }
        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);
        char msg[100];
        std::sprintf(msg, "Image id : %ld, Frame id: %ld, STATE: %d", ni, SLAM.GetCurrentFid(),SLAM.GetTrackingState());
        std::cout << msg << std::endl;
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();


        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];
        

        if(ttrack<T)
            usleep(0.8*(T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    std::sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(std::size_t ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    std::cout << "-------" << endl << endl;
    std::cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    std::cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   
    makedir("../tmp/");
    SLAM.SaveKeyFrames("../tmp/KeyFrames");
    SLAM.SaveMap("../tmp/Map.xml");
    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "times.txt";
    if(!file_exist(strPathTimeFile)){
        throw std::runtime_error("Timestamp file: "+strPathTimeFile+" does not exsit.");
        return;
    };  // time.txt must exist
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "image_0/";
    
    if(!file_exist(strPrefixLeft)){
        throw std::runtime_error("Image Directory: "+strPrefixLeft+" does not exsit.");
        return;
    };  // time.txt must exist
    listdir(vstrImageFilenames, strPrefixLeft);
    for(auto &imageFilename: vstrImageFilenames)
        imageFilename = strPrefixLeft + imageFilename;
    if(vTimestamps.size()!=vstrImageFilenames.size()){
        char msg[100];
        std::sprintf(msg, "size of vTimestamps (%ld) != size of vstrImageFilenames (%ld)", vTimestamps.size(), vstrImageFilenames.size());
        throw std::runtime_error(msg);
        return;
    }; 
}
