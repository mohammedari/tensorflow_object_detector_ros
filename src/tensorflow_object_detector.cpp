#include <sstream>
#include <stdexcept>

#include "tensorflow_object_detector.h"

TensorFlowObjectDetector::TensorFlowObjectDetector(const std::string& graph_path, const std::string& labels_path)
{
//    //load graph
//    {
//        auto status = TF_NewStatus();
//
//        graph_ = TF_NewGraph();
//
//        //import
//        auto graph_def = read_file_to_new_buffer(graph_path);
//        auto options = TF_NewImportGraphDefOptions();
//        TF_GraphImportGraphDef(graph_, graph_def, opions, status)
//        TF_DeleteImportGraphDefOptions(options);
//        TF_DeleteBuffer(graph_def);
//
//        throw_if_error(status, "Failed to load graph");
//
//        TF_DeleteStatus(status);
//    }
//
//    //create session
//    {
//        auto status = TF_NewStatus();
//
//        auto options = TF_NewSessionOptions();
//        session_ = TF_NewSession(graph_, options_, status_);
//        TF_DeleteSessionOptions(options)
//
//        throw_if_error(status, "Failed to create session");
//
//        TF_DeleteStatus(status)
//    }
//
//    //read labels
//    {
//        std::ifstream file(labels_path);
//        if (!file) {
//            std::stringstream ss;
//            ss << "Labels file " << labels_path << " is not found." << std::endl;
//            throw std::invalid_argument(ss.str());
//        }
//        std::string line;
//        while (std::getline(file, line)) {
//            labels_.push_back(line);
//        }
//    }
}

TensorFlowObjectDetector::~TensorFlowObjectDetector()
{
}

std::vector<TensorFlowObjectDetector::Result> TensorFlowObjectDetector::detect(const Eigen::Tensor<float, 3>& input_tensor, float score_threshold)
{
//    std::vector<tensorflow::Tensor> outputs;
//    auto run_status = session_->Run({{IMAGE_TENSOR, input_tensor}},
//                                     {DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, NUM_DETECTIONS}, {}, &outputs);
//    if (!run_status.ok()) {
//        std::stringstream ss;
//        ss << "Failed to run interference model: " << run_status.ToString() << std::endl;
//        throw std::runtime_error(ss.str());
//    }
//
//    const auto boxes_tensor = outputs[0].shaped<float, 2>({100, 4});     //shape={1, 100, 4}
//    const auto scores_tensor = outputs[1].shaped<float, 1>({100});       //shape={1, 100}
//    const auto classes_tensor = outputs[2].shaped<float, 1>({100});      //shape={1, 100}
//    const auto num_detections_tensor = outputs[3].shaped<float, 1>({1}); //shape={1}

    //retrieve and format valid results
    std::vector<Result> results;
//    for(int i = 0; i < num_detections_tensor(0); ++i) {
//        const float score = scores_tensor(i);
//        if (score < score_threshold) {
//            continue;
//        }
//
//        const Eigen::AlignedBox2f box(
//            Eigen::Vector2f(boxes_tensor(i, 1), boxes_tensor(i, 0)),
//            Eigen::Vector2f(boxes_tensor(i, 3), boxes_tensor(i, 2))
//        );
//        const int label_index = classes_tensor(i);
//
//        std::string label;
//        if (label_index <= labels_.size()) {
//            label = labels_[label_index - 1];
//        } else {
//            label = "unknown";
//        }
//        
//        std::stringstream ss;
//        ss << classes_tensor(i) << " : " << label;
//
//        results.push_back({
//            box, score, label_index, ss.str()
//        });
//    }

    return results;
}
