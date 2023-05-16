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
            if(!(*it)->isBad())
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
            if(!(*it)->isBad())
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

void Map::RestoreMapPointsConnection(cv::FileStorage &fs, unordered_map<int, KeyFrame*> &mapKFId, unordered_map<int, MapPoint*> &mapMptId)
{
    cv::FileNode mptsfn = fs["mspMapPoints"];
    for(cv::FileNodeIterator NodeIt = mptsfn.begin(); NodeIt != mptsfn.end(); ++NodeIt)  // MapPoint FileNode
    {
        vector<int> mobsMapKFId, mMapKFInId;
        int MptId;
        (*NodeIt)["mobsMapKFId"] >> mobsMapKFId;
        (*NodeIt)["mMapKFInId"] >> mMapKFInId;
        (*NodeIt)["mnId"] >> MptId;
        assert(mobsMapKFId.size() == mMapKFInId.size());
        auto MptIt = mapMptId.find(MptId); // verify the MapPoint
        if(MptIt == mapMptId.end()){
            std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mLost MapPoint in Hash Table\033[0m" << std::endl;
                continue;
        }
        MapPoint* pMpt = mapMptId[MptId]; // retrive the valid MapPoint
        for(int i = 0; i < (int) mobsMapKFId.size(); ++i){
            int KFId = mobsMapKFId[i], KFInId = mMapKFInId[i];
            auto KFIt = mapKFId.find(KFId);   // verify the KeyFrame
            if(KFIt == mapKFId.end()){
                std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mUnconnected MapPoint for KeyFrame\033[0m" << std::endl;
                continue;
            }
            if(KFInId > (int)(KFIt->second->mvKeysUn.size()-1) || KFInId < 0){
                std::cout << "[Warning] " <<__FILE__ << " Line " << __LINE__ <<": \033[33;1mIndex of Connected MapPoint Overflow the size of Keypoints\033[0m" << std::endl;
                continue;
            }
            KFIt->second->mmapMpt2Kpt[pMpt] = KFInId;   
            if(!pMpt->IsInKeyFrame(KFIt->second))
                pMpt->AddObservationStatic(KFIt->second, (size_t)KFInId);    // why Segment Fault??
        }
    }
    fs.release();
}

} //namespace ORB_SLAM


// Reloaded operator<<() of Map
cv::FileStorage& operator<<(cv::FileStorage &fs, ORB_SLAM2::Map &map){
    fs << "mspMapPoints" << "{";
    vector<ORB_SLAM2::MapPoint*> vmpts = map.GetAllMapPoints();
    for(int i = 0; i < (int)vmpts.size();++i){
        char index[15];
        sprintf(index, "%06d", i);
        fs << "MapPoint_"+std::string(index) << "{";
        fs << *(vmpts[i]) << "}";
    }
    fs << "}";
    vector<ORB_SLAM2::KeyFrame*> vKFs = map.GetAllKeyFrames();
    vector<int> vKFIds;
    for(auto KF:vKFs)
        vKFIds.push_back(KF->mnId);
    fs  << "mspKeyFrameId" << vKFIds;
    return fs;
}

