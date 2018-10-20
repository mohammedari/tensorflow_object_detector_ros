#ifndef TENSORFLOW_OBJECT_DETECTOR_H_
#define TENSORFLOW_OBJECT_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include "Eigen/Geometry"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/c/c_api.h"

class TensorFlowObjectDetector
{
private:
    //constant tensor names for tensorflow object detection api
    const std::string IMAGE_TENSOR = "image_tensor:0";
    const std::string DETECTION_BOXES = "detection_boxes:0";
    const std::string DETECTION_SCORES = "detection_scores:0";
    const std::string DETECTION_CLASSES = "detection_classes:0";
    const std::string NUM_DETECTIONS = "num_detections:0";

    std::vector<std::string> labels_;
    TF_Graph *graph_;
    TF_Session *session_;

public:
    struct Result
    {
        Eigen::AlignedBox2f box;
        float score;
        int label_index;
        std::string label;
    };

    TensorFlowObjectDetector(const std::string& graph_path, const std::string& labels_path);
    ~TensorFlowObjectDetector();

    std::vector<Result> detect(const Eigen::Tensor<float, 3>& image_tensor, float score_threshold);

    TensorFlowObjectDetector(const TensorFlowObjectDetector&) = delete;
    TensorFlowObjectDetector(TensorFlowObjectDetector&&) = delete;
    TensorFlowObjectDetector& operator=(const TensorFlowObjectDetector&) = delete;
    TensorFlowObjectDetector& operator=(TensorFlowObjectDetector&&) = delete;
};

#endif //TENSORFLOW_OBJECT_DETECTOR_H_