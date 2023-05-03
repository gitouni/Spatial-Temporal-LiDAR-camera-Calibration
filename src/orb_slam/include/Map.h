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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();
    friend class boost::serialization::access; // for serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        std::vector<MapPoint> mvMapPoints;
        for(auto it=mspMapPoints.begin(); it!=mspMapPoints.end();++it){
            mvMapPoints.push_back(**it);
        }
        ar& mvMapPoints;

    }
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames(bool onlyGood=false);
    std::vector<MapPoint*> GetAllMapPoints(bool onlyGood=false);
    std::vector<MapPoint*> GetReferenceMapPoints();
    static bool lId(MapPoint* pMpt1, MapPoint* pMpt2);
    long unsigned int MapPointsInMap();
    long unsigned KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    std::vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;
    void RestoreMap(cv::FileStorage &fs, std::vector<int> &vkFId);
    void RestoreMapPointsConnection(cv::FileStorage &fs, std::unordered_map<int, KeyFrame*> &mapKFId, unordered_map<int, MapPoint*> &mapMptId);

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM
cv::FileStorage& operator<<(cv::FileStorage &fs, ORB_SLAM2::Map &map);
#endif // MAP_H
