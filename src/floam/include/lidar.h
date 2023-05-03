// Author of FLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _LIDAR_H_
#define _LIDAR_H_

//define lidar parameter

namespace lidar{

class Lidar
{
    public:
        Lidar();

        void setScanPeriod(double scan_period_in);
        void setLines(short num_lines_in);
        //by default is 100. pls do not change
        void setMaxDistance(double max_distance_in);
        void setMinDistance(double min_distance_in);

    	double max_distance;
        double min_distance;
        int num_lines;
        double scan_period;
};


}


#endif // _LIDAR_H_

