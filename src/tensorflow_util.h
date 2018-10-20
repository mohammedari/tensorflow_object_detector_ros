#ifndef TENSORFLOW_UTIL_H_
#define TENSORFLOW_UTIL_H_

#include <memory>
#include <string>
#include <iostream>

#include "tensorflow/c/c_api.h"

class TensorFlowUtil
{
public:
    static std::unique_ptr<TF_Status> createStatus();
    static std::unique_ptr<TF_Buffer> createBuffer();
    static std::unique_ptr<TF_Buffer> createBuffer(const std::string& filename);

    static std::unique_ptr<TF_Session> createSession(TF_Graph* graph, const TF_SessionOptions* options);
    static std::unique_ptr<TF_SessionOptions> createSessionOptions();

    static std::unique_ptr<TF_Graph> createGraph();
    static std::unique_ptr<TF_ImportGraphDefOptions> createImportGraphDefOptions();

    template<size_t ROWS, size_t COLS, size_t CHANNEL>
    static std::unique_ptr<TF_Tensor> createTensor(void*);

    static void throwIfError(TF_Status* status, const std::string& message);

    TensorFlowUtil() = delete;
    ~TensorFlowUtil() = delete;
    TensorFlowUtil(const TensorFlowUtil&) = delete;
    TensorFlowUtil(TensorFlowUtil&&) = delete;
    TensorFlowUtil& operator=(const TensorFlowUtil&) = delete;
    TensorFlowUtil& operator=(TensorFlowUtil&&) = delete;
};

namespace std
{
    template<> struct default_delete<TF_Status>
    {
        void operator()(TF_Status* status) const
        {
            TF_DeleteStatus(status);
        }
    };

    template<> struct default_delete<TF_Buffer>
    {
        void operator()(TF_Buffer* buffer) const
        {
            TF_DeleteBuffer(buffer);
        }
    };

    template<> struct default_delete<TF_Session>
    {
        void operator()(TF_Session* session) const
        {
            auto status = TensorFlowUtil::createStatus();
            TF_DeleteSession(session, status.get());

            //This code does not throw error while deleting
            if (TF_OK != TF_GetCode(status.get()))
            {
                std::cerr
                    << "An error reported while deleting session: "
                    << TF_Message(status.get())
                    << std::endl;
            }
        }
    };

    template<> struct default_delete<TF_SessionOptions>
    {
        void operator()(TF_SessionOptions* options) const
        {
            TF_DeleteSessionOptions(options);
        }
    };

    template<> struct default_delete<TF_Graph>
    {
        void operator()(TF_Graph* graph) const
        {
            TF_DeleteGraph(graph);
        }
    };

    template<> struct default_delete<TF_ImportGraphDefOptions>
    {
        void operator()(TF_ImportGraphDefOptions* options) const
        {
            TF_DeleteImportGraphDefOptions(options);
        }
    };
    
    template<> struct default_delete<TF_Tensor>
    {
        void operator()(TF_Tensor* tensor) const
        {
            TF_DeleteTensor(tensor);
        }
    };
}

template<size_t ROWS, size_t COLS, size_t CHANNEL>
std::unique_ptr<TF_Tensor> TensorFlowUtil::createTensor(void*)
{
    //TODO
}

#endif //TENSORFLOW_UTIL_H_