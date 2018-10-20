#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "tensorflow_object_detector_nodecore.h"

TensorFlowObjectDetectorNodeCore::TensorFlowObjectDetectorNodeCore(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh_private), it_(nh) 
{
    //subscribers
    imageSubscriber_ = it_.subscribe("image_in", 1, &TensorFlowObjectDetectorNodeCore::imageCallback, this);

    //publishers
    imagePublisher_ = it_.advertise("image_out", 1);

    //params
    std::string graph_path, labels_path;
    nh_.param<std::string>("graph_path", graph_path, "");
    nh_.param<std::string>("labels_path", labels_path, "");
    nh_.param<double>("score_threshold", score_threshold_, 0.8);
    nh_.param<bool>("always_output_image", always_output_image_, false);

    //initialize tensorflow
    detector_.reset(new TensorFlowObjectDetector(graph_path, labels_path));
}

void TensorFlowObjectDetectorNodeCore::imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); //for tensorflow, using RGB8
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
        return;
    }

    // perform actual detection
    std::vector<TensorFlowObjectDetector::Result> results;
    {
        try
        {
            results = detector_->detect(cv_ptr->image, score_threshold_);
        }
        catch (std::runtime_error& e)
        {
            ROS_ERROR_STREAM("TensorFlow runtime detection error: " << e.what());
            return;
        }
    }

    // image output
    {
        cv_bridge::CvImage outImage;;
        outImage.header = cv_ptr->header;
        outImage.encoding = cv_ptr->encoding;
        outImage.image = cv_ptr->image.clone();

        auto height = cv_ptr->image.rows;
        auto width = cv_ptr->image.cols;

        //draw detection rect
        for (const auto& result : results)
        {
            static const cv::Scalar color(255, 0, 0);

            auto topleft = result.box.min();
            auto sizes = result.box.sizes();
            cv::Rect rect(topleft.x() * width, topleft.y() * height, sizes.x() * width, sizes.y() * height);

            // draw rectangle
            cv::rectangle(outImage.image, rect, color, 3);

            // draw label and score
            std::stringstream ss;
            ss << result.label << "(" << std::setprecision(2) << result.score << ")";
            cv::putText(
                outImage.image,
                ss.str(),
                cv::Point(
                    std::min(static_cast<int>(topleft.x() * width), 640 - 100),
                    std::max(static_cast<int>(topleft.y() * height - 10), 20)),
                cv::FONT_HERSHEY_PLAIN, 1.0, color);

        }

        if (always_output_image_ || results.size() > 0)
        {
            imagePublisher_.publish(outImage.toImageMsg());
        }
    }
}