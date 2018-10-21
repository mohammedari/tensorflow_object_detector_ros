#ifndef TENSORFLOW_OBJECT_DETECTOR_H_
#define TENSORFLOW_OBJECT_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "Eigen/Geometry"
#include "unsupported/Eigen/CXX11/Tensor"

#include "tensorflow_util.h"
#include "tensorflow/c/c_api.h"

class TensorFlowObjectDetector
{
private:
    //constant tensor names for tensorflow object detection api
    const std::string IMAGE_TENSOR = "image_tensor";
    const std::string DETECTION_BOXES = "detection_boxes";
    const std::string DETECTION_SCORES = "detection_scores";
    const std::string DETECTION_CLASSES = "detection_classes";
    const std::string NUM_DETECTIONS = "num_detections";

    std::vector<std::string> labels_;
    std::unique_ptr<TF_Graph> graph_;
    std::unique_ptr<TF_Session> session_;

    TF_Output image_tensor_;
    TF_Output detection_boxes_;
    TF_Output detection_scores_;
    TF_Output detection_classes_;
    TF_Output num_detections_;

public:
    struct Result
    {
        Eigen::AlignedBox2f box;
        float score;
        int label_index;
        std::string label;
    };

    TensorFlowObjectDetector(const std::string& graph_path, const std::string& labels_path);
    ~TensorFlowObjectDetector() = default;

    std::vector<Result> detect(const cv::Mat& image, float score_threshold);

    TensorFlowObjectDetector(const TensorFlowObjectDetector&) = delete;
    TensorFlowObjectDetector(TensorFlowObjectDetector&&) = delete;
    TensorFlowObjectDetector& operator=(const TensorFlowObjectDetector&) = delete;
    TensorFlowObjectDetector& operator=(TensorFlowObjectDetector&&) = delete;
};

#endif //TENSORFLOW_OBJECT_DETECTOR_H_