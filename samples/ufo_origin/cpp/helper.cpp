#include "helper.h"
#include "face_tracking_raw.h"

namespace SecureMR {

bool Helper::LoadModelData(const std::string& filePath, std::vector<char>& modelData) {
  if (g_assetManager == nullptr) {
    Log::Write(Log::Level::Error, "AssetManager is not set");
    return false;
  }

  AAsset* asset = AAssetManager_open(g_assetManager, filePath.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    Log::Write(Log::Level::Error, Fmt("Open MR: Error: Failed to open asset %s", filePath.c_str()));
    return false;
  }

  off_t assetLength = AAsset_getLength(asset);
  modelData.resize(assetLength);

  AAsset_read(asset, modelData.data(), assetLength);
  AAsset_close(asset);

  return true;
}

XrSecureMrTensorPICO Helper::CreateTensorAsSlice(const XrSecureMrFrameworkPICO& framework,
                                                 const std::vector<int>& start, const std::vector<int>& end,
                                                 const std::vector<int>& skip, int32_t dimension, int32_t sliceSize) {
  if (start.size() != end.size()) {
    throw std::invalid_argument("Start and end sizes must be equal");
  }
  if (!skip.empty() && skip.size() != start.size()) {
    throw std::invalid_argument("Skip must be empty or the same size as start and end.");
  }

  int8_t channelSize = skip.empty() ? 2 : 3;
  XrSecureMrTensorFormatPICO sliceTensorFormat = {XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, channelSize,
                                                  XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO};

  int32_t dimensions[] = {dimension};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo = {
      XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO, nullptr, false, 1, dimensions, &sliceTensorFormat};

  XrSecureMrTensorPICO sliceTensor;
  CHECK_XRCMD(xrCreateSecureMrTensorPICO(
      framework, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &sliceTensor));

  XrSecureMrTensorBufferPICO tensorBuffer = {};
  tensorBuffer.type = XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO;
  tensorBuffer.next = nullptr;
  tensorBuffer.bufferSize = sliceSize;
  tensorBuffer.buffer = malloc(tensorBuffer.bufferSize);
  if (tensorBuffer.buffer == nullptr) {
    throw std::runtime_error("Failed to allocate memory for slice tensor buffer.");
  }

  // Fill the buffer with `start`, `end`, and optional `skip` values
  int32_t* bufferData = static_cast<int32_t*>(tensorBuffer.buffer);
  for (size_t i = 0; i < start.size(); ++i) {
    bufferData[channelSize * i] = static_cast<int32_t>(start[i]);
    bufferData[channelSize * i + 1] = static_cast<int32_t>(end[i]);
    if (channelSize == 3) {
      bufferData[channelSize * i + 2] = static_cast<int32_t>(skip[i]);  // Only include skip if provided
    }
  }
  CHECK_XRCMD(xrResetSecureMrTensorPICO(sliceTensor, &tensorBuffer));
  return sliceTensor;
}

XrSecureMrPipelineTensorPICO Helper::CreatePipelineTensorAsSlice(const XrSecureMrPipelinePICO& pipeline,
                                                                 const std::vector<int>& start,
                                                                 const std::vector<int>& end,
                                                                 const std::vector<int>& skip, int32_t dimension,
                                                                 int32_t sliceSize) {
  if (start.size() != end.size()) {
    throw std::invalid_argument("Start and end sizes must be equal");
  }
  if (!skip.empty() && skip.size() != start.size()) {
    throw std::invalid_argument("Skip must be empty or the same size as start and end.");
  }

  int8_t channelSize = skip.empty() ? 2 : 3;
  XrSecureMrTensorFormatPICO sliceTensorFormat = {XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, channelSize,
                                                  XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO};

  int32_t dimensions[] = {dimension};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo = {
      XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO, nullptr, false, 1, dimensions, &sliceTensorFormat};

  XrSecureMrPipelineTensorPICO sliceTensor;
  CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
      pipeline, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &sliceTensor));

  XrSecureMrTensorBufferPICO tensorBuffer = {};
  tensorBuffer.type = XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO;
  tensorBuffer.next = nullptr;
  tensorBuffer.bufferSize = sliceSize;
  tensorBuffer.buffer = malloc(tensorBuffer.bufferSize);
  if (tensorBuffer.buffer == nullptr) {
    throw std::runtime_error("Failed to allocate memory for slice tensor buffer.");
  }

  // Fill the buffer with `start`, `end`, and optional `skip` values
  int32_t* bufferData = static_cast<int32_t*>(tensorBuffer.buffer);
  for (size_t i = 0; i < start.size(); ++i) {
    bufferData[channelSize * i] = static_cast<int32_t>(start[i]);
    bufferData[channelSize * i + 1] = static_cast<int32_t>(end[i]);
    if (channelSize == 3) {
      bufferData[channelSize * i + 2] = static_cast<int32_t>(skip[i]);  // Only include skip if provided
    }
  }
  CHECK_XRCMD(xrResetSecureMrPipelineTensorPICO(pipeline, sliceTensor, &tensorBuffer));
  return sliceTensor;
}

void Helper::InitializePipelineIOPair(XrSecureMrPipelineIOPairPICO* pair, XrSecureMrPipelineTensorPICO placeholder,
                                      XrSecureMrTensorPICO tensor) {
  pair->type = XR_TYPE_SECURE_MR_PIPELINE_IO_PAIR_PICO;
  pair->next = nullptr;
  pair->localPlaceHolderTensor = placeholder;
  pair->globalTensor = tensor;
}

void Helper::CreateAndSetGlobalTensor(XrSecureMrFrameworkPICO framework, XrSecureMrTensorPICO& tensor,
                                      const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                      XrSecureMrTensorDataTypePICO tensorDataType, XrSecureMrTensorTypePICO tensorType,
                                      const void* buffer, size_t bufferSize) {
  XrSecureMrTensorFormatPICO tensorFormat{tensorDataType, channel, tensorType};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                       nullptr,
                                                       false,
                                                       dimensionsCount,
                                                       const_cast<int32_t*>(dimensions),
                                                       &tensorFormat};
  CHECK_XRCMD(xrCreateSecureMrTensorPICO(
      framework, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tensor));

  XrSecureMrTensorBufferPICO tensorBuffer{XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO, nullptr,
                                          static_cast<uint32_t>(bufferSize), const_cast<void*>(buffer)};
  CHECK_XRCMD(xrResetSecureMrTensorPICO(tensor, &tensorBuffer));
}

void Helper::CreateAndSetPipelineTensor(XrSecureMrPipelinePICO pipeline, XrSecureMrPipelineTensorPICO& tensor,
                                        const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                        XrSecureMrTensorDataTypePICO tensorDataType,
                                        XrSecureMrTensorTypePICO tensorType, const void* buffer, size_t bufferSize,
                                        bool isPlaceholder) {
  XrSecureMrTensorFormatPICO tensorFormat{tensorDataType, channel, tensorType};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                       nullptr,
                                                       isPlaceholder,
                                                       dimensionsCount,
                                                       const_cast<int32_t*>(dimensions),
                                                       &tensorFormat};
  CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
      pipeline, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tensor));

  XrSecureMrTensorBufferPICO tensorBuffer{XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO, nullptr,
                                          static_cast<uint32_t>(bufferSize), const_cast<void*>(buffer)};
  CHECK_XRCMD(xrResetSecureMrPipelineTensorPICO(pipeline, tensor, &tensorBuffer));
}

void Helper::CreateGlobalTensor(XrSecureMrFrameworkPICO framework, XrSecureMrTensorPICO& tensor,
                                const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                XrSecureMrTensorDataTypePICO tensorDataType, XrSecureMrTensorTypePICO tensorType,
                                bool isPlaceholder) {
  XrSecureMrTensorFormatPICO tensorFormat{tensorDataType, static_cast<int8_t>(channel), tensorType};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                       nullptr,
                                                       isPlaceholder,
                                                       static_cast<uint32_t>(dimensionsCount),
                                                       const_cast<int32_t*>(dimensions),
                                                       &tensorFormat};

  CHECK_XRCMD(xrCreateSecureMrTensorPICO(
      framework, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tensor));
}

void Helper::CreatePipelineTensor(XrSecureMrPipelinePICO pipeline, XrSecureMrPipelineTensorPICO& tensor,
                                  const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                  XrSecureMrTensorDataTypePICO tensorDataType, XrSecureMrTensorTypePICO tensorType,
                                  bool isPlaceholder) {
  XrSecureMrTensorFormatPICO tensorFormat{tensorDataType, static_cast<int8_t>(channel), tensorType};
  XrSecureMrTensorCreateInfoShapePICO tensorCreateInfo{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                       nullptr,
                                                       isPlaceholder,
                                                       static_cast<uint32_t>(dimensionsCount),
                                                       const_cast<int32_t*>(dimensions),
                                                       &tensorFormat};

  CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
      pipeline, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tensor));
}

void Helper::CreateOperator(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO& op,
                            XrSecureMrOperatorTypePICO operatorType) {
  XrSecureMrOperatorBaseHeaderPICO operatorInfo{XR_TYPE_SECURE_MR_OPERATOR_BASE_HEADER_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
                                                      &operatorInfo, operatorType};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(pipeline, &operatorCreateInfo, &op));
}

void Helper::CreateArithmeticOperator(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO& op, const char* config) {
  XrSecureMrOperatorArithmeticComposePICO arithmeticOperatorPico{
      XR_TYPE_SECURE_MR_OPERATOR_ARITHMETIC_COMPOSE_PICO, nullptr, {}  // Zero-initialize configText
  };

  // Ensure we don't overflow the buffer
  strncpy(arithmeticOperatorPico.configText, config, sizeof(arithmeticOperatorPico.configText) - 1);
  arithmeticOperatorPico.configText[sizeof(arithmeticOperatorPico.configText) - 1] = '\0';  // Null-terminate

  XrSecureMrOperatorCreateInfoPICO arithmeticCreateInfoPico{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&arithmeticOperatorPico),
      XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(pipeline, &arithmeticCreateInfoPico, &op));
}

void Helper::SetInput(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO op,
                      XrSecureMrPipelineTensorPICO inputTensor, const char* name) {
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(pipeline, op, inputTensor, name));
}

void Helper::SetOutput(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO op,
                       XrSecureMrPipelineTensorPICO outputTensor, const char* name) {
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(pipeline, op, outputTensor, name));
}

std::shared_ptr<Helper> CreateHelper(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<Helper>(instance, session);
}

};  // namespace SecureMR
