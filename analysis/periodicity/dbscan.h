#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"
#include "../mapData_struct.h"

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;



typedef struct Point_
{
    float x, y;  // X, Y, Z position
    int clusterID;  // clustered ID
}Point;


struct DBscan_Settings
{
    numb eps;			// Initial deflection
    int analysedVariable;	// Variable to analyse
    numb CoefIntervals;
    numb CoefPeaks;


    __device__ DBscan_Settings(numb _eps, int _analysedVariable, numb _CoefIntervals, numb _CoefPeaks)
    {
        CoefIntervals = _CoefIntervals;
        CoefPeaks = _CoefPeaks;
        eps = _eps;
        analysedVariable = _analysedVariable;
    }

};

class DBSCAN {
public:    
    DBSCAN(unsigned int minPts, float eps, vector<Point> points){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
        clusterCount = 0;
    }
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(Point point);
    int expandCluster(Point point, int clusterID);
    inline double calculateDistance(const Point& pointCore, const Point& pointTarget);

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    
public:
    vector<Point> m_points;
    int clusterCount;
    
private:    
    
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

__device__ void Periodicity(Computation* data, DBscan_Settings settings, int variation, int offset);

#endif // DBSCAN_H
