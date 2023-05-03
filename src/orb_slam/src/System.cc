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



#include "System.h"
#include "Converter.h"
#include "Optimizer.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <fstream>
#include <omp.h>
using namespace std;
namespace ORB_SLAM2
{

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}


cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}
void System::GetKeyFramePoses(std::vector<cv::Mat> &vKFTwc, std::vector<std::size_t> &vKFFrameId){
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames(true);
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
    for(size_t i=0;i<vpKFs.size();i++)
    {
        vKFTwc.push_back(vpKFs[i]->GetPoseInverse());
        vKFFrameId.push_back(vpKFs[i]->mnFrameId);
    }
}
//add by chy
int System::OptimizeExtrinsicLocal(const vector<Eigen::Isometry3d> &vTwl, Eigen::Isometry3d &Tcl, double &scale, bool verbose)
{
    vector<KeyFrame*> vgoodKF=mpMap->GetAllKeyFrames(true);
    sort(vgoodKF.begin(), vgoodKF.end(), KeyFrame::lId);
    return Optimizer::OptimizeExtrinsicLocal(vgoodKF, vTwl, Tcl, scale, verbose);
}
int System::OptimizeExtrinsicGlobal(const vector<Eigen::Isometry3d> &vTwl, Eigen::Isometry3d &Tcl, double &scale, bool verbose)
{
    vector<KeyFrame*> vgoodKF=mpMap->GetAllKeyFrames(true);
    sort(vgoodKF.begin(), vgoodKF.end(), KeyFrame::lId);
    return Optimizer::OptimizeExtrinsicGlobal(vgoodKF, vTwl, Tcl, scale, verbose);
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

size_t System::GetLastLoopKFid()
{
    return mpLoopCloser->GetLastLoopKFid();
}

size_t System::GetCurrentFid()
{
    return mpTracker->GetCurrentFid();
}
// Reloaded operator<<()
void System::SaveMap(const std::string &filename)
{
    std::cout << "Saving the Map..." << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << *mpMap;
    std::cout << "Map Saved." << std::endl;
}

void System::SaveKeyFrames(const string &dirname)
{
    std::string KeyFrameDir(dirname);
    checkpath(KeyFrameDir);
    makedir(KeyFrameDir.c_str());
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames(true);
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    vector<MapPoint*> vpMappts = mpMap->GetAllMapPoints(true);
    sort(vpMappts.begin(), vpMappts.end(), Map::lId);
    unordered_map<KeyFrame*, int32_t> mapKFId;
    unordered_map<MapPoint*, int32_t> mapMptId;
    for(int32_t i =0 ;i < (int32_t)vpKFs.size(); ++i)
        mapKFId.insert(std::make_pair(vpKFs[i], vpKFs[i]->mnId));
        
    for(int32_t i = 0; i < (int32_t)vpMappts.size(); ++i)
        mapMptId.insert(std::make_pair(vpMappts[i], vpMappts[i]->mnId));
    std::cout << "Saving KeyFrames..." << std::endl;
    for(int32_t i = 0; i < (int32_t)vpKFs.size(); ++i){
        char basename[15];
        sprintf(basename, "%06d", i);
        string fileName(basename);
        fileName = KeyFrameDir + fileName;
        vpKFs[i]->saveData(fileName, mapKFId, mapMptId);
        if(vpKFs.size() > 100)  // May wait some time. A simple Process Indicator is added.
        {
            if((i+1) % (vpKFs.size() / 10) == 0)
                std::cout << 100.0*(i+1)/((double)vpKFs.size()) << "% Saved." << std::endl;
        }
        
    }
    std::cout << vpKFs.size() << " KeyFrames saved to " + KeyFrameDir << "." << std::endl;
}

void System::RestoreSystemFromFile(const string &keyFrameDir, const string &mapFilename){
    string kFdirname(keyFrameDir);
    checkpath(kFdirname);
    assert(file_exist(kFdirname));
    assert(file_exist(mapFilename));
    cv::FileStorage fs(mapFilename, cv::FileStorage::READ);
    std::cout << "Files are valid. Start to restore the ORB_SLAM System..." << std::endl;
    vector<int> KFinMapId;
    mpMap->RestoreMap(fs, KFinMapId);
    vector<MapPoint*> goodMapPoints = mpMap->GetAllMapPoints(true);
    std::cout << "Found " << goodMapPoints.size() << " MapPoints." << std::endl;
    unordered_map<int, MapPoint*> mapMapPointId;
    mapMapPointId.reserve(goodMapPoints.size());
    for(MapPoint* mpt : goodMapPoints)
        mapMapPointId[mpt->mnId] = mpt;
    std::cout << "" << goodMapPoints.size() << " MapPoints Restored." << std::endl;
    vector<string> InfoFile, FeatFile;
    DIR *dirp = nullptr;
    struct dirent *dir_entry = nullptr;
    if((dirp=opendir(kFdirname.c_str()))==nullptr){
        throw std::runtime_error("Cannot open directory: "+kFdirname);
    }
    while((dir_entry=readdir(dirp))!=nullptr){
        if(dir_entry->d_type!=DT_REG)
            continue;
        string filename = dir_entry->d_name;
        string suffix = filename.substr(filename.find_last_of('.') + 1);
        if(suffix=="bin")
            FeatFile.push_back(kFdirname+filename);
        if(suffix=="yml" || suffix=="yaml")
            InfoFile.push_back(kFdirname+filename);
    }
    closedir(dirp); 
    assert(FeatFile.size()==InfoFile.size());
    std::cout << "Found " << FeatFile.size() << " KeyFrames." << std::endl;
    vector<KeyFrameConstInfo*> vConstInfo;
    vector<KeyFrame*> vpKF;
    unordered_map<int, KeyFrame*> mapKeyFrameId;
    vConstInfo.resize(FeatFile.size());
    vpKF.resize(FeatFile.size());
    mapKeyFrameId.reserve(FeatFile.size());
    int cnt = 0;
    #pragma omp parallel for ordered schedule(dynamic)      // open it for multithread reading, but sometimes it causes failed points
    for(std::size_t fi = 0; fi < FeatFile.size(); ++ fi)
    {
        string featFilename = FeatFile[fi], infoFilename = InfoFile[fi];
        cv::FileStorage cvfs(infoFilename, cv::FileStorage::READ);
        ifstream ifs(featFilename, ios::binary);
        boost::archive::binary_iarchive iar(ifs);
        KeyFrameConstInfo* constInfo(new KeyFrameConstInfo(cvfs, mpKeyFrameDatabase, mpVocabulary));
        KeyFrame* newKF(new KeyFrame(std::ref(iar), constInfo, mpMap, goodMapPoints, mapMapPointId));

    #pragma omp ordered
    {
        vConstInfo[fi] = constInfo;
        vpKF[fi] = newKF;
        if(FeatFile.size() > 100)  // May wait some time. A simple Process Indicator is added.
        {
            ++ cnt;
            if((cnt) % (FeatFile.size() / 10) == 0)
                std::cout << 100.0*(cnt+1)/((double)FeatFile.size()) << "% Loaded." << std::endl;
        }
    }
    }
    std::cout << vpKF.size() << " KeyFrames Restored." << std::endl;
    // May be unnecessary to sort
    std::sort(vpKF.begin(),vpKF.end(),KeyFrame::lId);
    std::sort(vConstInfo.begin(),vConstInfo.end(),KeyFrameConstInfo::lId);
    for(KeyFrame* pKF:vpKF)
        mapKeyFrameId[pKF->mnId] = pKF;
    for(int kFi = 0; kFi < (int) vpKF.size(); kFi++)
        vpKF[kFi]->GlobalConnection(vConstInfo[kFi], mapKeyFrameId);
    std::cout << "KeyFrame-to-KeyFrame Connections Restored." << std::endl;
    cv::FileStorage refs(mapFilename, cv::FileStorage::READ);
    mpMap->RestoreMapPointsConnection(refs, mapKeyFrameId, mapMapPointId);
    std::cout << "MapPoint-to-KeyFrame Connections Restored." << std::endl;
    for(KeyFrame* pKF: vpKF)
        mpMap->AddKeyFrame(pKF);
    for(KeyFrameConstInfo* ConstInfo: vConstInfo){
        delete ConstInfo;
        (void)ConstInfo;
    }
}

void System::makedir(const char* folder){
    if(access(folder,F_OK)){
        if(mkdir(folder,0755)!=0)
            printf("\033[1;31mDirectory %s created Failed with unknown error!\033[0m\n",folder);
        else
            printf("\033[1;32mDirectory %s created successfully!\033[0m\n",folder);
    }else
        printf("\033[1;33mDirectory %s not accessible or alreadly exists!\033[0m\n",folder);
}


} //namespace ORB_SLAM
