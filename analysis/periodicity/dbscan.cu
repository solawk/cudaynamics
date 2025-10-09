#include "dbscan.h"


__device__  void Period(Computation* data, DBscan_Settings settings, int variation,  int offset) {
    int variationStart = variation * CUDA_marshal.variationSize;
    numb eps = settings.eps;
    int analysedVariable = settings.analysedVariable;
    numb coefPeaks = settings.CoefPeaks;
    numb coefIntervals = settings.CoefIntervals;

    int varCount = CUDA_kernel.VAR_COUNT;
    int variationSize = CUDA_marshal.variationSize;

    // Buffer to hold peak data (amplitudes and indices)
    constexpr int MAX_PEAKS = 512;
    numb peakAmplitudes[MAX_PEAKS];
    numb peakIntervals[MAX_PEAKS];

    int peakCount = 0;
    numb tempPeakAmp = 0, tempPeakTime = 0; bool tempPeakFound = false;
    bool firstpeakreached = false;
    numb temppeakindex;
    numb* computedVariation = CUDA_marshal.trajectory + variationStart;
    for (int i = 1; i < variationSize / varCount - 1 && peakCount < MAX_PEAKS; i++)
    {
        numb prev = computedVariation[analysedVariable + varCount * i - varCount];
        numb curr = computedVariation[analysedVariable + varCount * i];
        numb next = computedVariation[analysedVariable + varCount * i + varCount];
        if (curr > prev && curr > next)
        {
            tempPeakFound = false;
            if (firstpeakreached == false)
            {
                firstpeakreached = true;
                temppeakindex = (float)i;
            }
            else
            {

                peakAmplitudes[peakCount] = curr;
                peakIntervals[peakCount] = (i - temppeakindex) * CUDA_kernel.stepSize;
                peakCount++;
                temppeakindex = (float)i;
            }
        }
        else if (curr == next && curr > prev) {
            tempPeakFound = true; tempPeakAmp = curr; tempPeakTime = i;
        }
        else if (curr < next) {
            tempPeakFound = false;
        }
        else if (curr > next && tempPeakFound) {
            if (firstpeakreached) {
                peakAmplitudes[peakCount] = tempPeakAmp;  peakIntervals[peakCount] = (tempPeakTime - temppeakindex) * CUDA_kernel.stepSize;
                peakCount++;
                temppeakindex = (float)tempPeakTime;
                tempPeakFound = false;
            }
            else {
                firstpeakreached = true;
                temppeakindex = (float)tempPeakTime;
                tempPeakFound = false;
            }
        }
    }
   
    for (int i = 0; i < peakCount-1; i++) {
        peakIntervals[i] *= coefIntervals; peakAmplitudes[i] *= coefPeaks;
    }

    int cluster = 0;
    int NumNeibor = 0;
    int helpfulArray[MAX_PEAKS];
    for (int i = 0; i < MAX_PEAKS; ++i) {
        helpfulArray[i] = 0;
    }

     for (int i = 0; i < peakCount; i++)
        if (NumNeibor >= 1)
        {
            i = helpfulArray[peakCount + NumNeibor - 1];
            helpfulArray[peakCount + NumNeibor - 1] = 0;
            NumNeibor = NumNeibor - 1;
            for (int k = 0; k < peakCount - 1; k++) {
                if (i != k && helpfulArray[k] == 0) {
                    if (sqrt(pow(peakAmplitudes[i] - peakAmplitudes[k], 2) + pow(peakIntervals[i] - peakIntervals[k], 2)) <= eps) {
                        helpfulArray[k] = cluster;
                        helpfulArray[peakCount + k] = k;
                        ++NumNeibor;
                    }
                }

            }
        }
        else if (helpfulArray[i] == 0) {
            NumNeibor = 0;
            ++cluster;
            helpfulArray[i] = cluster;
            for (int k = 0; k < peakCount - 1; k++) {
                if (i != k && helpfulArray[peakCount + k] == 0) {
                    if (sqrt(pow(peakAmplitudes[i] - peakAmplitudes[k], 2) + pow(peakIntervals[i] - peakIntervals[k], 2)) <= eps) {
                        helpfulArray[k] = cluster;
                        helpfulArray[peakCount + k] = k;
                        ++NumNeibor;
                    }
                }

            }
        }  
     cluster--;

     numb mapValue = cluster;

     if (CUDA_kernel.mapWeight == 0.0f)
     {
         numb existingValue = CUDA_marshal.maps[mapPosition] * data->bufferNo;
         CUDA_marshal.maps[mapPosition] = (existingValue + mapValue) / (data->bufferNo + 1);
     }
     else if (CUDA_kernel.mapWeight == 1.0f)
     {
         CUDA_marshal.maps[mapPosition] = mapValue;
     }
     else
     {
         CUDA_marshal.maps[mapPosition] = CUDA_marshal.maps[mapPosition] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
     }
    
}

int DBSCAN::run()
{
    int clusterID = 1;
    vector<Point>::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1; clusterCount++;
            }
        }
    }
    for (iter = m_points.begin(); iter != m_points.end(); ++iter) {
        if (iter->clusterID == NOISE || iter->clusterID == FAILURE || iter->clusterID == UNCLASSIFIED) {
            clusterCount++;
        }
    }

    return 0;
}

int DBSCAN::expandCluster(Point point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(Point point)
{
    int index = 0;
    vector<Point>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(const Point& pointCore, const Point& pointTarget )
{
    return pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2);
}



