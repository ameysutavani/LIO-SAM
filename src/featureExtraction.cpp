#include "lio_sam/cloud_info.h"
#include "utility.h"

struct smoothness_t
{
  float value;
  size_t ind;
};

struct by_value
{
  bool operator()(smoothness_t const &left, smoothness_t const &right)
  {
    return left.value < right.value;
  }
};

class FeatureExtraction : public ParamServer
{
public:
  ros::Subscriber
      subLaserCloudInfo; ///< Subscriber for output of imageProjection
                         ///< (Contains deskewed cloud and necessary structure
                         ///< information)

  ros::Publisher
      pubLaserCloudInfo; ///< Publisher for extracted features. cornerPoints
                         ///< and surfacePoint indices are filled
  ros::Publisher
      pubCornerPoints;             ///< Publisher for visualizing extracted corner features
  ros::Publisher pubSurfacePoints; ///< Publisher for visualizing extracted
                                   ///< surface features

  pcl::PointCloud<PointType>::Ptr
      extractedCloud; ///< Pointer used to reference current input deskewed
                      ///< cloud
  pcl::PointCloud<PointType>::Ptr
      cornerCloud; ///< Pointer used to reference current extracted corner
                   ///< cloud
  pcl::PointCloud<PointType>::Ptr
      surfaceCloud; ///< Pointer used to reference current extracted surface
                    ///< cloud

  pcl::VoxelGrid<PointType>
      downSizeFilter; ///< VoxelGrid filter for surface points

  lio_sam::cloud_info
      cloudInfo;                ///< This maintains a copy in current input cloudInfo message
  std_msgs::Header cloudHeader; ///< This maintains a copy in current input
                                ///< laser cloudInfor message header

  std::vector<smoothness_t>
      cloudSmoothness; ///< A container to hold a copy of smoothness and flat
                       ///< index of each point. In each section,
                       ///< corressponding parts of this container are sorted
                       ///< according the smoothness value; smoothest points
                       ///< are extracted (labelled at at associated index) as
                       ///< surface features, sharpest points are extracted as
                       ///< corner features

  float *cloudCurvature; ///< Float array holding computed smoothness per point

  int *cloudNeighborPicked; ///< (Boolean) Mask denoting if the point is either
                            ///< 1. Occluded OR
                            ///< 2. Sampelled by parallel beams OR
                            ///< 3. Is a neighbor of already picked feature point. These points are
                            ///< skipped in feature extraction.

  int *cloudLabel; ///< Denotes (feature) label of each point:
                   ///< -1: Surface feature point
                   ///< 0: Non-feature(skipped) point
                   ///< 1: Corner feature point

  FeatureExtraction()
  {
    subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/deskew/cloud_info", 1,
        &FeatureExtraction::laserCloudInfoHandler, this,
        ros::TransportHints().tcpNoDelay());

    pubLaserCloudInfo =
        nh.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1);
    pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/feature/cloud_corner", 1);
    pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/feature/cloud_surface", 1);

    initializationValue();
  }

  void initializationValue()
  {
    cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

    downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize,
                               odometrySurfLeafSize);

    extractedCloud.reset(new pcl::PointCloud<PointType>());
    cornerCloud.reset(new pcl::PointCloud<PointType>());
    surfaceCloud.reset(new pcl::PointCloud<PointType>());

    cloudCurvature = new float[N_SCAN * Horizon_SCAN];
    cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
    cloudLabel = new int[N_SCAN * Horizon_SCAN];
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr &msgIn)
  {
    cloudInfo = *msgIn;          // new cloud info
    cloudHeader = msgIn->header; // new cloud header
    pcl::fromROSMsg(msgIn->cloud_deskewed,
                    *extractedCloud); // new cloud for extraction

    calculateSmoothness(); ///< IMP! Smoothness is calculated ignoring occlusion

    markOccludedPoints();

    extractFeatures();

    publishFeatureCloud();
  }

  void calculateSmoothness()
  {
    int cloudSize = extractedCloud->points.size();
    for (int i = 5; i < cloudSize - 5; i++) ///< IMP! Smoothness filter is convolved ignoring scan line
                                            ///< indices. This essentially is putting an assumption that
                                            ///< previous scan line ends close to where new scan line
                                            ///< startes. This wont hold true for non-360 pointclouds.
    {
      float diffRange =
          cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] +
          cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] +
          cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 +
          cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] +
          cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] +
          cloudInfo.pointRange[i + 5];

      cloudCurvature[i] =
          diffRange *
          diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

      cloudNeighborPicked[i] = 0;
      cloudLabel[i] = 0;
      // cloudSmoothness for sorting
      cloudSmoothness[i].value = cloudCurvature[i];
      cloudSmoothness[i].ind = i;
    }
  }

  void markOccludedPoints()
  {
    int cloudSize = extractedCloud->points.size();
    // mark occluded points and parallel beam points
    for (int i = 5; i < cloudSize - 6; ++i)
    {
      // occluded points
      float depth1 = cloudInfo.pointRange[i];
      float depth2 = cloudInfo.pointRange[i + 1];
      int columnDiff = std::abs(
          int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

      if (columnDiff < 10)
      {
        // 10 pixel diff in range image
        if (depth1 - depth2 > 0.3)
        {
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
        else if (depth2 - depth1 > 0.3)
        {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
      // parallel beam
      float diff1 = std::abs(
          float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
      float diff2 = std::abs(
          float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

      if (diff1 > 0.02 * cloudInfo.pointRange[i] &&
          diff2 > 0.02 * cloudInfo.pointRange[i])
        cloudNeighborPicked[i] = 1;
    }
  }

  void extractFeatures()
  {
    cornerCloud->clear();
    surfaceCloud->clear();

    pcl::PointCloud<PointType>::Ptr surfaceCloudScan(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(
        new pcl::PointCloud<PointType>());

    for (int i = 0; i < N_SCAN; i++) ///< Here, i denotes the current scan line index
    {
      surfaceCloudScan->clear();

      for (int j = 0; j < 6; j++) ///< Here, j denotes a (azimuth) section index
                                  ///< in a scan line. In this case, each scan
                                  ///< line is divided into 6 sections (0->5)
      {
        int sp = (cloudInfo.startRingIndex[i] * (6 - j) +
                  cloudInfo.endRingIndex[i] * j) /
                 6; ///< This is just computing flattened starting index of current section in current scan line
        int ep = (cloudInfo.startRingIndex[i] * (5 - j) +
                  cloudInfo.endRingIndex[i] * (j + 1)) /
                     6 -
                 1; ///< This is just computing flattened ending index of current section in current scan line

        if (sp >= ep)
          continue;

        std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,
                  by_value()); ///< Sort current section's points in ascending
        ///< order according to their "sharpness" value

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--)
        {
          int ind = cloudSmoothness[k].ind;
          if (cloudNeighborPicked[ind] == 0 &&
              cloudCurvature[ind] > edgeThreshold)
          {
            largestPickedNum++;
            if (largestPickedNum <= 20)
            {
              cloudLabel[ind] = 1;
              cornerCloud->push_back(extractedCloud->points[ind]);
            }
            else
            {
              break;
            }

            cloudNeighborPicked[ind] = 1;
            for (int l = 1; l <= 5; l++)
            {
              int columnDiff =
                  std::abs(int(cloudInfo.pointColInd[ind + l] -
                               cloudInfo.pointColInd[ind + l - 1]));
              if (columnDiff > 10)
                break;
              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--)
            {
              int columnDiff =
                  std::abs(int(cloudInfo.pointColInd[ind + l] -
                               cloudInfo.pointColInd[ind + l + 1]));
              if (columnDiff > 10)
                break;
              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++)
        {
          int ind = cloudSmoothness[k].ind;
          if (cloudNeighborPicked[ind] == 0 &&
              cloudCurvature[ind] < surfThreshold)
          {
            cloudLabel[ind] = -1;
            cloudNeighborPicked[ind] = 1;

            for (int l = 1; l <= 5; l++)
            {
              int columnDiff =
                  std::abs(int(cloudInfo.pointColInd[ind + l] -
                               cloudInfo.pointColInd[ind + l - 1]));
              if (columnDiff > 10)
                break;

              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--)
            {
              int columnDiff =
                  std::abs(int(cloudInfo.pointColInd[ind + l] -
                               cloudInfo.pointColInd[ind + l + 1]));
              if (columnDiff > 10)
                break;

              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++)
        {
          if (cloudLabel[k] <= 0)
          {
            surfaceCloudScan->push_back(extractedCloud->points[k]);
          }
        }
      }

      surfaceCloudScanDS->clear();
      downSizeFilter.setInputCloud(surfaceCloudScan);
      downSizeFilter.filter(*surfaceCloudScanDS);

      *surfaceCloud += *surfaceCloudScanDS;
    }
  }

  void freeCloudInfoMemory()
  {
    cloudInfo.startRingIndex.clear();
    cloudInfo.endRingIndex.clear();
    cloudInfo.pointColInd.clear();
    cloudInfo.pointRange.clear();
  }

  void publishFeatureCloud()
  {
    // free cloud info memory
    freeCloudInfoMemory();
    // save newly extracted features
    cloudInfo.cloud_corner = publishCloud(&pubCornerPoints, cornerCloud,
                                          cloudHeader.stamp, lidarFrame);
    cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud,
                                           cloudHeader.stamp, lidarFrame);
    // publish to mapOptimization
    pubLaserCloudInfo.publish(cloudInfo);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lio_sam");

  FeatureExtraction FE;

  ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

  ros::spin();

  return 0;
}