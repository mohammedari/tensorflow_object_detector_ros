#include "tensorflow_object_detector_nodecore.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorflow_object_detector");

  TensorFlowObjectDetectorNodeCore node(ros::NodeHandle(), ros::NodeHandle("~"));
  node.run();
}

