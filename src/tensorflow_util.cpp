#include "tensorflow_util.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::unique_ptr<TF_Status> TensorFlowUtil::createStatus()
{
    return std::unique_ptr<TF_Status>(TF_NewStatus());
}

std::unique_ptr<TF_Buffer> TensorFlowUtil::createBuffer()
{
    return std::unique_ptr<TF_Buffer>(TF_NewBuffer());
}

std::unique_ptr<TF_Buffer> TensorFlowUtil::createBuffer(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    auto data = new char[size];
    if (!file.read(data, size))
    {
        std::stringstream ss;
        ss << "Could not read file: " << filename;
        throw std::runtime_error(ss.str());
    }

    static const auto deleter = [](void* data, size_t size)
    {
        delete[] static_cast<char*>(data);
    };

    auto buf = createBuffer();
    buf->data = data;
    buf->length = size;
    buf->data_deallocator = deleter;

    return buf;
}

std::unique_ptr<TF_Session> TensorFlowUtil::createSession(TF_Graph* graph, const TF_SessionOptions* options)
{
    auto status = TensorFlowUtil::createStatus();
    auto ptr = std::unique_ptr<TF_Session>(TF_NewSession(graph, options, status.get()));

    throwIfError(status.get(), "An error reported while creating session");

    return ptr;
}

std::unique_ptr<TF_SessionOptions> TensorFlowUtil::createSessionOptions()
{
    return std::unique_ptr<TF_SessionOptions>(TF_NewSessionOptions());
}

std::unique_ptr<TF_Graph> TensorFlowUtil::createGraph()
{
    return std::unique_ptr<TF_Graph>(TF_NewGraph());
}

std::unique_ptr<TF_ImportGraphDefOptions> TensorFlowUtil::createImportGraphDefOptions()
{
    return std::unique_ptr<TF_ImportGraphDefOptions>(TF_NewImportGraphDefOptions());
}

void TensorFlowUtil::throwIfError(TF_Status* status, const std::string& message)
{
    if (TF_OK != TF_GetCode(status))
    {
        std::stringstream ss;
        ss << message << ": " << TF_Message(status);
        throw std::runtime_error(ss.str());
    }
}
