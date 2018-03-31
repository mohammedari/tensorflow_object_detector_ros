#ifndef TENSORFLOW_OBJECT_DETECTOR_H_
#define TENSORFLOW_OBJECT_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include "Eigen/Geometry"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

class TensorflowObjectDetector
{
private:
    //constant tensor names for tensorflow object detection api
    const std::string IMAGE_TENSOR = "image_tensor:0";
    const std::string DETECTION_BOXES = "detection_boxes:0";
    const std::string DETECTION_SCORES = "detection_scores:0";
    const std::string DETECTION_CLASSES = "detection_classes:0";
    const std::string NUM_DETECTIONS = "num_detections:0";

    std::vector<std::string> labels_;
    std::unique_ptr<tensorflow::Session> session_;

public:
    struct Result
    {
        Eigen::AlignedBox2f box;
        float score;
        int label_index;
        std::string label;
    };

    TensorflowObjectDetector(const std::string& graph_path, const std::string& labels_path);

    //TODO tensorflow::Tensorを直接渡さずに、Eigen::Tensorなどで画像を渡したい
    std::vector<Result> detect(const tensorflow::Tensor& image_tensor, float score_threshold);
};

#endif //TENSORFLOW_OBJECT_DETECTOR_H_