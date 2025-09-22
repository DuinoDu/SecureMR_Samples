#include "mnistwild.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <chrono>
#include <filesystem>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "securemr_utils/serialization.h"

extern AAssetManager* g_assetManager;
extern std::string g_internalDataPath;

using json = SecureMR::Json;

namespace SecureMR {
namespace {
constexpr std::array<float, 6> kCropSrcPoints{1444.0F, 1332.0F, 2045.0F, 1332.0F, 2045.0F, 1933.0F};
constexpr std::array<float, 6> kCropDstPoints{0.0F, 0.0F, static_cast<float>(MnistWildApp::kCropWidth), 0.0F,
                                              static_cast<float>(MnistWildApp::kCropWidth),
                                              static_cast<float>(MnistWildApp::kCropHeight)};
constexpr int kCvColorRgb2Gray = 7;  // Matches cv::COLOR_RGB2GRAY

constexpr char kInferencePipelineJson[] = "mnist_inference_pipeline.json";
constexpr char kTensorPredictedClass[] = "predicted_class";
constexpr char kTensorPredictedScore[] = "predicted_score";
constexpr char kTensorCropImage[] = "cropped_image";
constexpr char kTensorRightEye[] = "right_eye_uint8";
constexpr char kTensorLeftEye[] = "left_eye_uint8";
constexpr char kTensorTimestamp[] = "timestamp_tensor";
constexpr char kTensorCameraMatrix[] = "camera_matrix_tensor";
constexpr char kTensorAffine[] = "affine_tensor";
constexpr char kTensorCropRgb[] = "crop_rgb_tensor";
constexpr char kTensorCropGray[] = "crop_gray_tensor";
constexpr char kTensorCropFloat[] = "crop_float_tensor";
constexpr char kTensorNormalized[] = "normalized_input_tensor";

std::filesystem::path ResolveWritablePath(const std::string& fileName) {
  if (!g_internalDataPath.empty()) {
    return std::filesystem::path(g_internalDataPath) / fileName;
  }
  return {};
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, float value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(float));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, uint16_t value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(uint16_t));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, uint8_t value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(uint8_t));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakePointTensor(const std::shared_ptr<Pipeline>& pipeline, const std::array<float, 2>& p) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 2,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<float*>(p.data())), sizeof(float) * 2);
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeColorTensor(const std::shared_ptr<Pipeline>& pipeline,
                                                const std::array<uint8_t, 8>& rgbaPair) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {2},
                                .channels = 4,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<uint8_t*>(rgbaPair.data())), rgbaPair.size());
  return tensor;
}

std::shared_ptr<PipelineTensor> MakePoseTensor(const std::shared_ptr<Pipeline>& pipeline,
                                               const std::array<float, 16>& mat) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {4, 4},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<float*>(mat.data())), sizeof(float) * mat.size());
  return tensor;
}
}  // namespace

MnistWildApp::MnistWildApp(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

MnistWildApp::~MnistWildApp() {
  keepRunning = false;
  initCv.notify_all();
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) {
      runner.join();
    }
  }
}

bool MnistWildApp::LoadAsset(const std::string& filePath, std::vector<char>& data) const {
  if (g_assetManager == nullptr) {
    Log::Write(Log::Level::Error, "LoadAsset failed: AssetManager not available");
    return false;
  }
  AAsset* asset = AAssetManager_open(g_assetManager, filePath.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    Log::Write(Log::Level::Error, Fmt("LoadAsset failed: unable to open %s", filePath.c_str()));
    return false;
  }
  const off_t length = AAsset_getLength(asset);
  data.resize(static_cast<size_t>(length));
  const int64_t read = AAsset_read(asset, data.data(), length);
  AAsset_close(asset);
  if (read != length) {
    Log::Write(Log::Level::Error, Fmt("LoadAsset failed: read %ld of %ld bytes", static_cast<long>(read),
                                      static_cast<long>(length)));
    data.clear();
    return false;
  }
  return true;
}

bool MnistWildApp::DeserializeInferencePipeline(const std::filesystem::path& jsonPath) {
  json spec = LoadJsonFromFile(jsonPath);
  PipelineDeserializationResult result;
  PipelineDeserializationOptions options;
  options.customOperatorHandler = [this](const json& opSpec,
                                         const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor,
                                         const std::shared_ptr<Pipeline>& pipeline,
                                         std::string& error) -> bool {
    const std::string type = opSpec.value("type", "");
    if (type != "run_algorithm") {
      return false;
    }
    if (mnistModelBuffer.empty()) {
      error = "run_algorithm requires loaded model buffer";
      return false;
    }
    auto mappedInputs = ParseMappedTensorList(opSpec.value("inputs", json::array()));
    auto mappedOutputs = ParseMappedTensorList(opSpec.value("outputs", json::array()));
    if (mappedInputs.empty() || mappedOutputs.empty()) {
      error = "run_algorithm inputs/outputs malformed";
      return false;
    }
    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> inputMap;
    for (const auto& [alias, tensorName] : mappedInputs) {
      inputMap.emplace(alias, requireTensor(tensorName));
    }
    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> outputMap;
    for (const auto& [alias, tensorName] : mappedOutputs) {
      outputMap.emplace(alias, requireTensor(tensorName));
    }
    const std::string modelName = opSpec.value("model_name", "mnist");
    std::unordered_map<std::string, std::string> aliasing;
    pipeline->runAlgorithm(mnistModelBuffer.data(), mnistModelBuffer.size(), inputMap, aliasing, outputMap, aliasing,
                           modelName);
    return true;
  };

  std::string error;
  if (!DeserializePipelineFromJson(spec, frameworkSession, result, error, options)) {
    Log::Write(Log::Level::Error,
              Fmt("DeserializeInferencePipeline failed: %s", error.empty() ? "unknown error" : error.c_str()));
    return false;
  }

  inferencePipeline = result.pipeline;
  try {
    predClassPlaceholder = result.tensorMap.at(kTensorPredictedClass);
    predScorePlaceholder = result.tensorMap.at(kTensorPredictedScore);
    cropImagePlaceholder = result.tensorMap.at(kTensorCropImage);
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error,
              Fmt("DeserializeInferencePipeline failed: required placeholder missing (%s)", e.what()));
    return false;
  }
  return true;
}

void MnistWildApp::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, kImageWidth, kImageHeight);
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void MnistWildApp::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    CreateGlobalTensors();
    CreateInferencePipeline();
    CreateRenderPipeline();
    {
      std::lock_guard<std::mutex> lk(initMutex);
      pipelinesReady = true;
    }
    initCv.notify_all();
  });
}

void MnistWildApp::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    std::unique_lock<std::mutex> lk(initMutex);
    initCv.wait(lk, [this]() { return pipelinesReady || !keepRunning; });
    lk.unlock();
    while (keepRunning) {
      RunInferencePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    std::unique_lock<std::mutex> lk(initMutex);
    initCv.wait(lk, [this]() { return pipelinesReady || !keepRunning; });
    lk.unlock();
    while (keepRunning) {
      RunRenderPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
  });
}

void MnistWildApp::CreateGlobalTensors() {
  Log::Write(Log::Level::Info, "Creating global tensors ...");

  predictedClassGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  int32_t defaultClass = -1;
  predictedClassGlobal->setData(reinterpret_cast<int8_t*>(&defaultClass), sizeof(defaultClass));

  predictedScoreGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float zeroScore = 0.0F;
  predictedScoreGlobal->setData(reinterpret_cast<int8_t*>(&zeroScore), sizeof(zeroScore));

  croppedImageGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  std::vector<uint8_t> blankImage(static_cast<size_t>(kCropHeight * kCropWidth * 3), 0);
  croppedImageGlobal->setData(reinterpret_cast<int8_t*>(blankImage.data()), blankImage.size());

  std::vector<char> gltfData;
  if (LoadAsset("tv.gltf", gltfData)) {
    gltfClassAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfScoreAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfImageAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
  } else {
    Log::Write(Log::Level::Error, "Failed to load tv.gltf");
  }

  if (!LoadAsset("mnist.serialized.bin", mnistModelBuffer)) {
    Log::Write(Log::Level::Error, "Failed to load mnist.serialized.bin");
  }

  Log::Write(Log::Level::Info, "Global tensors ready.");
}

void MnistWildApp::CreateInferencePipeline() {
  Log::Write(Log::Level::Info, "Creating inference pipeline ...");
#ifdef LOAD_FROM_JSON_ONLY
  const std::filesystem::path jsonPath = ResolveWritablePath(kInferencePipelineJson);
  if (!DeserializeInferencePipeline(jsonPath)) {
    const std::string pathStr = jsonPath.empty() ? "<empty path>" : jsonPath.string();
    Log::Write(Log::Level::Error,
              Fmt("Failed to load inference pipeline from %s", pathStr.c_str()));
  }
#else
  inferencePipeline = std::make_shared<Pipeline>(frameworkSession);

  predClassPlaceholder = PipelineTensor::PipelinePlaceholderLike(inferencePipeline, predictedClassGlobal);
  predScorePlaceholder = PipelineTensor::PipelinePlaceholderLike(inferencePipeline, predictedScoreGlobal);
  cropImagePlaceholder = PipelineTensor::PipelinePlaceholderLike(inferencePipeline, croppedImageGlobal);

  auto rightEyeUint8 = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kImageHeight, kImageWidth},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto leftEyeUint8 = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kImageHeight, kImageWidth},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto timestampTensor =
      std::make_shared<PipelineTensor>(inferencePipeline, static_cast<TensorAttribute>(TensorAttribute_TimeStamp{}));
  auto cameraMatrixTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {3, 3},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto affineTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {2, 3},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto cropRgbTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto cropGrayTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto cropFloatTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto normalizedInputTensor = std::make_shared<PipelineTensor>(
      inferencePipeline,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*inferencePipeline)
      .cameraAccess(rightEyeUint8, leftEyeUint8, timestampTensor, cameraMatrixTensor)
      .getAffine(kCropSrcPoints, kCropDstPoints, affineTensor)
      .applyAffine(affineTensor, leftEyeUint8, cropRgbTensor)
      .assignment(cropRgbTensor, cropImagePlaceholder)
      .cvtColor(kCvColorRgb2Gray, cropRgbTensor, cropGrayTensor)
      .typeConvert(cropGrayTensor, cropFloatTensor)
      .arithmetic("({0} / 255.0)", {cropFloatTensor}, normalizedInputTensor);

  if (!mnistModelBuffer.empty()) {
    (*inferencePipeline)
        .runAlgorithm(mnistModelBuffer.data(), mnistModelBuffer.size(), {{"input_1", normalizedInputTensor}}, {},
                      {{"_538", predScorePlaceholder}, {"_539", predClassPlaceholder}}, {}, "mnist");
  } else {
    Log::Write(Log::Level::Error, "Skip model inference: model buffer empty");
  }

  std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> tensorSpecMap = {
      {kTensorPredictedClass, predClassPlaceholder},
      {kTensorPredictedScore, predScorePlaceholder},
      {kTensorCropImage, cropImagePlaceholder},
      {kTensorRightEye, rightEyeUint8},
      {kTensorLeftEye, leftEyeUint8},
      {kTensorTimestamp, timestampTensor},
      {kTensorCameraMatrix, cameraMatrixTensor},
      {kTensorAffine, affineTensor},
      {kTensorCropRgb, cropRgbTensor},
      {kTensorCropGray, cropGrayTensor},
      {kTensorCropFloat, cropFloatTensor},
      {kTensorNormalized, normalizedInputTensor},
  };
  const std::unordered_set<std::string> placeholderNames = {kTensorPredictedClass, kTensorPredictedScore,
                                                            kTensorCropImage};

  json spec;
  spec["metadata"]["version"] = 1;
  json tensorsJson = json::object();
  for (const auto& entry : tensorSpecMap) {
    json tensorJson =
        TensorAttributeVariantToJson(entry.second ? entry.second->getAttribute() : std::variant<std::monostate, TensorAttribute>{});
    tensorJson["is_placeholder"] = placeholderNames.count(entry.first) > 0;
    tensorsJson[entry.first] = tensorJson;
  }
  spec["tensors"] = tensorsJson;

  json operators = json::array();

  json cameraOp;
  cameraOp["type"] = "camera_access";
  cameraOp["inputs"] = json::array();
  cameraOp["outputs"] = TensorListToJson(
      {kTensorRightEye, kTensorLeftEye, kTensorTimestamp, kTensorCameraMatrix});
  operators.push_back(cameraOp);

  json getAffineOp;
  getAffineOp["type"] = "get_affine";
  json srcPoints = json::array();
  for (float v : kCropSrcPoints) {
    srcPoints.push_back(v);
  }
  json dstPoints = json::array();
  for (float v : kCropDstPoints) {
    dstPoints.push_back(v);
  }
  getAffineOp["src_points"] = srcPoints;
  getAffineOp["dst_points"] = dstPoints;
  getAffineOp["inputs"] = json::array();
  getAffineOp["outputs"] = TensorListToJson({kTensorAffine});
  operators.push_back(getAffineOp);

  json applyAffineOp;
  applyAffineOp["type"] = "apply_affine";
  applyAffineOp["inputs"] = TensorListToJson({kTensorAffine, kTensorLeftEye});
  applyAffineOp["outputs"] = TensorListToJson({kTensorCropRgb});
  operators.push_back(applyAffineOp);

  json assignOp;
  assignOp["type"] = "assignment";
  assignOp["inputs"] = TensorListToJson({kTensorCropRgb});
  assignOp["outputs"] = TensorListToJson({kTensorCropImage});
  operators.push_back(assignOp);

  json cvtColorOp;
  cvtColorOp["type"] = "cvt_color";
  cvtColorOp["flag"] = kCvColorRgb2Gray;
  cvtColorOp["inputs"] = TensorListToJson({kTensorCropRgb});
  cvtColorOp["outputs"] = TensorListToJson({kTensorCropGray});
  operators.push_back(cvtColorOp);

  json typeConvertOp;
  typeConvertOp["type"] = "type_convert";
  typeConvertOp["inputs"] = TensorListToJson({kTensorCropGray});
  typeConvertOp["outputs"] = TensorListToJson({kTensorCropFloat});
  operators.push_back(typeConvertOp);

  json arithmeticOp;
  arithmeticOp["type"] = "arithmetic";
  arithmeticOp["expression"] = "({0} / 255.0)";
  arithmeticOp["inputs"] = TensorListToJson({kTensorCropFloat});
  arithmeticOp["outputs"] = TensorListToJson({kTensorNormalized});
  operators.push_back(arithmeticOp);

  if (!mnistModelBuffer.empty()) {
    json runAlg;
    runAlg["type"] = "run_algorithm";
    runAlg["model_name"] = "mnist";
    runAlg["model_asset"] = "mnist.serialized.bin";
    runAlg["inputs"] = MappedTensorListToJson({{"input_1", kTensorNormalized}});
    runAlg["outputs"] =
        MappedTensorListToJson({{"_538", kTensorPredictedScore}, {"_539", kTensorPredictedClass}});
    operators.push_back(runAlg);
  }

  spec["operators"] = operators;

  const std::filesystem::path jsonPath = ResolveWritablePath(kInferencePipelineJson);
  if (WriteJsonToFile(jsonPath, spec)) {
    if (DeserializeInferencePipeline(jsonPath)) {
      Log::Write(Log::Level::Info,
                 Fmt("Inference pipeline restored from %s", jsonPath.string().c_str()));
    }
  }
#endif  // LOAD_FROM_JSON_ONLY

  Log::Write(Log::Level::Info, "Inference pipeline ready.");
}

void MnistWildApp::CreateRenderPipeline() {
  Log::Write(Log::Level::Info, "Creating render pipeline ...");
  renderPipeline = std::make_shared<Pipeline>(frameworkSession);

  renderClassPlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, predictedClassGlobal);
  renderScorePlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, predictedScoreGlobal);
  renderCropPlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, croppedImageGlobal);
  renderClassGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);
  renderScoreGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);
  renderImageGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);

  auto digitTextStart = MakePointTensor(renderPipeline, {0.1F, 0.3F});
  auto scoreTextStart = MakePointTensor(renderPipeline, {0.1F, 0.3F});
  auto textColors = MakeColorTensor(renderPipeline, {255, 255, 255, 255, 0, 0, 0, 255});
  auto textTextureIdClass = MakeScalarTensor(renderPipeline, static_cast<uint16_t>(0));
  auto textTextureIdScore = MakeScalarTensor(renderPipeline, static_cast<uint16_t>(0));
  auto fontSizeDigit = MakeScalarTensor(renderPipeline, 144.0F);
  auto fontSizeScore = MakeScalarTensor(renderPipeline, 144.0F);

  auto newTextureId = std::make_shared<PipelineTensor>(
      renderPipeline,
      TensorAttribute{.dimensions = {1},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});

  auto classPose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, -0.5F,
                                   0.0F, 0.5F, 0.0F, 0.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto scorePose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, 0.5F,
                                   0.0F, 0.5F, 0.0F, 0.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto imagePose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, 0.0F,
                                   0.0F, 0.5F, 0.0F, 1.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto visibleTensor = MakeScalarTensor(renderPipeline, static_cast<uint8_t>(1));

  (*renderPipeline)
      .execRenderCommand(std::make_shared<RenderCommand_DrawText>(renderClassGltfPlaceholder, "en-US",
                                                                  RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF,
                                                                  1440, 960, renderClassPlaceholder, digitTextStart,
                                                                  fontSizeDigit, textColors, textTextureIdClass))
      .execRenderCommand(std::make_shared<RenderCommand_DrawText>(renderScoreGltfPlaceholder, "en-US",
                                                                  RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF,
                                                                  1440, 960, renderScorePlaceholder, scoreTextStart,
                                                                  fontSizeScore, textColors, textTextureIdScore))
      .newTextureToGLTF(renderImageGltfPlaceholder, renderCropPlaceholder, newTextureId);

  auto updateMaterialCmd = std::make_shared<RenderCommand_UpdateMaterial>();
  updateMaterialCmd->gltfTensor = renderImageGltfPlaceholder;
  updateMaterialCmd->materialIds = std::vector<uint16_t>{0};
  updateMaterialCmd->attribute = RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
  updateMaterialCmd->materialValues = newTextureId;
  (*renderPipeline).execRenderCommand(updateMaterialCmd);

  auto renderClassCmd = std::make_shared<RenderCommand_Render>(renderClassGltfPlaceholder, classPose);
  renderClassCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderClassCmd);

  auto renderScoreCmd = std::make_shared<RenderCommand_Render>(renderScoreGltfPlaceholder, scorePose);
  renderScoreCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderScoreCmd);

  auto renderImageCmd = std::make_shared<RenderCommand_Render>(renderImageGltfPlaceholder, imagePose);
  renderImageCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderImageCmd);

  Log::Write(Log::Level::Info, "Render pipeline ready.");
}

void MnistWildApp::RunInferencePipeline() {
  if (!inferencePipeline) {
    return;
  }
  inferencePipeline->submit({{predClassPlaceholder, predictedClassGlobal},
                             {predScorePlaceholder, predictedScoreGlobal},
                             {cropImagePlaceholder, croppedImageGlobal}},
                            XR_NULL_HANDLE, nullptr);
}

void MnistWildApp::RunRenderPipeline() {
  if (!renderPipeline || gltfClassAsset == nullptr || gltfScoreAsset == nullptr || gltfImageAsset == nullptr) {
    return;
  }
  renderPipeline->submit({{renderClassPlaceholder, predictedClassGlobal},
                          {renderScorePlaceholder, predictedScoreGlobal},
                          {renderCropPlaceholder, croppedImageGlobal},
                          {renderClassGltfPlaceholder, gltfClassAsset},
                          {renderScoreGltfPlaceholder, gltfScoreAsset},
                          {renderImageGltfPlaceholder, gltfImageAsset}},
                         XR_NULL_HANDLE, nullptr);
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<MnistWildApp>(instance, session);
}

}  // namespace SecureMR
