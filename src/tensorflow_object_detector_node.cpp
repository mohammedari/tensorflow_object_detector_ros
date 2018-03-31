#include "tensorflow_object_detector_nodecore.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorflow_object_detector");

  TensorflowObjectDetectorNodeCore node(ros::NodeHandle(), ros::NodeHandle("~"));
  node.run();
}

