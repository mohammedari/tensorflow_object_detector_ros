#ifndef TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_
#define TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

#include "tensorflow_object_detector.h"

class TensorFlowObjectDetectorNodeCore
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber imageSubscriber_;
    image_transport::Publisher imagePublisher_;

    std::unique_ptr<TensorFlowObjectDetector> detector_;

    //parameters
    double score_threshold_;
    bool always_output_image_;

public:
    TensorFlowObjectDetectorNodeCore(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    ~TensorFlowObjectDetectorNodeCore() = default;
    
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

    void run()
    {
        ros::spin();
    }

    TensorFlowObjectDetectorNodeCore(const TensorFlowObjectDetectorNodeCore&) = delete;
    TensorFlowObjectDetectorNodeCore(TensorFlowObjectDetectorNodeCore&&) = delete;
    TensorFlowObjectDetectorNodeCore& operator=(const TensorFlowObjectDetectorNodeCore&) = delete;
    TensorFlowObjectDetectorNodeCore& operator=(TensorFlowObjectDetectorNodeCore&&) = delete;
};

#endif //TENSORFLOW_OBJECT_DETECTOR_NODECORE_H_