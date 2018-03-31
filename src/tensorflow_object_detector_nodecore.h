#ifndef TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_
#define TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

#include "tensorflow_object_detector.h"

class TensorflowObjectDetectorNodeCore
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber imageSubscriber_;
    image_transport::Publisher imagePublisher_;

    std::unique_ptr<TensorflowObjectDetector> detector_;

    //parameters
    double score_threshold_;
    bool always_output_image_;

public:
    TensorflowObjectDetectorNodeCore(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

    void run()
    {
        ros::spin();
    }
};

#endif //TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_