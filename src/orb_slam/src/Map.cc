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

#include "Map.h"

#include<mutex>
using namespace std;
namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames(bool onlygood)
{
    unique_lock<mutex> lock(mMutexMap);
    if(!onlygood)
        return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
    else{
        vector<KeyFrame*> goodKeyFrames;
        goodKeyFrames.reserve(mspKeyFrames.size());
        for(auto it=mspKeyFrames.begin(); it != mspKeyFrames.end(); ++it){
            if((*it) == NULL)
                continue;
            if((*it)->isBad())
                continue;
            goodKeyFrames.push_back(*it);
        }
        return goodKeyFrames;
    }
}

vector<MapPoint*> Map::GetAllMapPoints(bool onlygood)
{
    unique_lock<mutex> lock(mMutexMap);
    if(!onlygood)
        return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
    else{
        vector<MapPoint*> goodMapPoints;
        goodMapPoints.reserve(mspMapPoints.size());
        for(auto it=mspMapPoints.begin(); it != mspMapPoints.end(); ++it){
            if((*it) == NULL)
                continue;
            if((*it)->isBad())
                continue;
            goodMapPoints.push_back(*it);
        }
        return goodMapPoints;
    }
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}
bool Map::lId(MapPoint* pMpt1, MapPoint* pMpt2)
{
    return pMpt1->mnId<pMpt2->mnId;
}
void Map::RestoreMap(cv::FileStorage &fs, vector<int> &vkFId)
{
    cv::FileNode mptsfn = fs["mspMapPoints"];
    for(cv::FileNodeIterator it = mptsfn.begin(); it != mptsfn.end(); ++it){
        AddMapPoint(new MapPoint(*it, this));
    }
    fs["mspKeyFrameId"] >> vkFId;
    fs.release();
}

void Map::RestoreMapPointsConnection(cv::FileStorage &fs, unordered_map<int, KeyFrame*> &KeyFrameQuery, unordered_map<int, MapPoint*> &MapPointQuery)
{
    cv::FileNode mptsfn = fs["mspMapPoints"];
    for(cv::FileNodeIterator NodeIt = mptsfn.begin(); NodeIt != mptsfn.end(); ++ NodeIt)  // MapPoint FileNode
    {
        vector<int> mobsMapKFId, mMapKFInId;
        int MptId;
        (*NodeIt)["mobsMapKFId"] >> mobsMapKFId; // id of KeyFrames by which this MapPoint is observed
        (*NodeIt)["mMapKFInId"] >> mMapKFInId; // id of Corresponding KeyPoints in above KeyFrames
        (*NodeIt)["mnId"] >> MptId; // id of this obeserved MapPoints
        assert(mobsMapKFId.size() == mMapKFInId.size());
        auto MptIt = MapPointQuery.find(MptId); // verify the MapPoint
        if(MptIt == MapPointQuery.end()){
            std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mLost MapPoint in Hash Table\033[0m" << std::endl;
                continue;
        }
        MapPoint* pMpt = MapPointQuery[MptId]; // retrive the valid MapPoint
        for(int i = 0; i < (int) mobsMapKFId.size(); ++i){
            const int KFId = mobsMapKFId[i], KFInId = mMapKFInId[i];
            auto KFIt = KeyFrameQuery.find(KFId);   
            if(KFIt == KeyFrameQuery.end()) // ensure the KeyFrame Id is in mobsMapKFId
            {
                std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mUnconnected MapPoint for KeyFrame\033[0m" << std::endl;
                continue;
            }
            if(KFInId > (int)(KFIt->second->mvKeysUn.size()-1) || KFInId < 0) // ensure the Keypoint Id is valid
            {
                std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mIndex of Connected MapPoint Overflow the size of Keypoints\033[0m" << std::endl;
                continue;
            }
            // KFIt->second->mmapMpt2Kpt[pMpt] = KFInId;   // mmapMpt2Kpt has been restored by KeyFrame()
            if(!pMpt->IsInKeyFrame(KFIt->second))
                pMpt->AddObservationStatic(KFIt->second, (std::size_t)KFInId);
        }
    }
    fs.release();
}

} //namespace ORB_SLAM


// Reloaded operator<<() of Map
cv::FileStorage& operator<<(cv::FileStorage &fs, ORB_SLAM2::Map &map){
    fs << "mspMapPoints" << "{";
    vector<ORB_SLAM2::MapPoint*> vmpts = map.GetAllMapPoints();
    for(ORB_SLAM2::MapPoint* mpt:vmpts){
        char index[15];
        sprintf(index, "%ld", mpt->mnId);
        fs << "MapPoint_"+std::string(index) << "{";
        fs << *(mpt) << "}";
    }
    fs << "}";
    vector<ORB_SLAM2::KeyFrame*> vKFs = map.GetAllKeyFrames();
    vector<int> vKFIds;
    for(auto KF:vKFs)
        vKFIds.push_back(KF->mnId);
    fs  << "mspKeyFrameId" << vKFIds;
    return fs;
}

