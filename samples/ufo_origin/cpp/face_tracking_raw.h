#pragma once
#include "pch.h"
#include <fstream>
#include <random>
#include "logger.h"
#include "common.h"
#include "securemr_base.h"
#include "helper.h"

#define FACE_DETECTION_MODEL_PATH "facedetector_fp16_qnn229.bin"
#define GLTF_PATH "UFO.gltf"
#define ANCHOR_MAT "anchors_1.mat"

namespace SecureMR {

#define GET_INTSANCE_PROC_ADDR(name, ptr) \
  CHECK_XRCMD(xrGetInstanceProcAddr(xr_instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&ptr)));

class FaceTrackingRaw : public ISecureMR {
 public:
  FaceTrackingRaw(const XrInstance& instance, const XrSession& session);

  ~FaceTrackingRaw() override;

  void CreateFramework() override;

  void CreatePipelines() override;

  void RunPipelines() override;

  [[nodiscard]] bool LoadingFinished() const override { return pipelineAllInitialized; }

 protected:
  void GetVSTImages();

  void RunModelInference();

  void Map2Dto3D();

  void CreateSecureMrVSTImagePipeline();

  void CreateSecureMrModelInferencePipeline();

  void CreateSecureMrMap2dTo3dPipeline();

  void CreateSecureMrRenderingPipeline();

  void RunSecureMrVSTImagePipeline();

  void RunSecureMrModelInferencePipeline();

  void RunSecureMrMap2dTo3dPipeline();

  void RunSecureMrRenderingPipeline();

 protected:
  void getInstanceProcAddr() {
    Log::Write(Log::Level::Info, "getInstanceProcAddr start.");
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrFrameworkPICO", xrCreateSecureMrFrameworkPICO)
    GET_INTSANCE_PROC_ADDR("xrDestroySecureMrFrameworkPICO", xrDestroySecureMrFrameworkPICO)
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrPipelinePICO", xrCreateSecureMrPipelinePICO)
    GET_INTSANCE_PROC_ADDR("xrDestroySecureMrPipelinePICO", xrDestroySecureMrPipelinePICO)
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrOperatorPICO", xrCreateSecureMrOperatorPICO)
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrTensorPICO", xrCreateSecureMrTensorPICO)
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrPipelineTensorPICO", xrCreateSecureMrPipelineTensorPICO)
    GET_INTSANCE_PROC_ADDR("xrResetSecureMrTensorPICO", xrResetSecureMrTensorPICO)
    GET_INTSANCE_PROC_ADDR("xrResetSecureMrPipelineTensorPICO", xrResetSecureMrPipelineTensorPICO)
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorOperandByNamePICO", xrSetSecureMrOperatorOperandByNamePICO)
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorOperandByIndexPICO", xrSetSecureMrOperatorOperandByIndexPICO)
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorResultByNamePICO", xrSetSecureMrOperatorResultByNamePICO)
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorResultByIndexPICO", xrSetSecureMrOperatorResultByIndexPICO)
    GET_INTSANCE_PROC_ADDR("xrExecuteSecureMrPipelinePICO", xrExecuteSecureMrPipelinePICO);
    Log::Write(Log::Level::Info, "getInstanceProcAddr end.");
  }

 private:
  XrInstance xr_instance;
  XrSession xr_session;
  PFN_xrCreateSecureMrFrameworkPICO xrCreateSecureMrFrameworkPICO = nullptr;
  PFN_xrDestroySecureMrFrameworkPICO xrDestroySecureMrFrameworkPICO = nullptr;
  PFN_xrCreateSecureMrPipelinePICO xrCreateSecureMrPipelinePICO = nullptr;
  PFN_xrDestroySecureMrPipelinePICO xrDestroySecureMrPipelinePICO = nullptr;
  PFN_xrCreateSecureMrOperatorPICO xrCreateSecureMrOperatorPICO = nullptr;
  PFN_xrCreateSecureMrTensorPICO xrCreateSecureMrTensorPICO = nullptr;
  PFN_xrCreateSecureMrPipelineTensorPICO xrCreateSecureMrPipelineTensorPICO = nullptr;
  PFN_xrResetSecureMrTensorPICO xrResetSecureMrTensorPICO = nullptr;
  PFN_xrResetSecureMrPipelineTensorPICO xrResetSecureMrPipelineTensorPICO = nullptr;
  PFN_xrSetSecureMrOperatorOperandByNamePICO xrSetSecureMrOperatorOperandByNamePICO = nullptr;
  PFN_xrSetSecureMrOperatorOperandByIndexPICO xrSetSecureMrOperatorOperandByIndexPICO = nullptr;
  PFN_xrExecuteSecureMrPipelinePICO xrExecuteSecureMrPipelinePICO = nullptr;
  PFN_xrSetSecureMrOperatorResultByNamePICO xrSetSecureMrOperatorResultByNamePICO = nullptr;
  PFN_xrSetSecureMrOperatorResultByIndexPICO xrSetSecureMrOperatorResultByIndexPICO = nullptr;

  XrSecureMrFrameworkPICO m_secureMrFramework{XR_NULL_HANDLE};
  XrSecureMrPipelinePICO m_secureMrVSTImagePipeline{XR_NULL_HANDLE};
  XrSecureMrPipelinePICO m_secureMrModelInferencePipeline{XR_NULL_HANDLE};
  XrSecureMrPipelinePICO m_secureMrMap2dTo3dPipeline{XR_NULL_HANDLE};
  XrSecureMrPipelinePICO m_secureMrRenderingPipeline{XR_NULL_HANDLE};

  // Pipeline IO
  XrSecureMrTensorPICO vstOutputLeftUint8{XR_NULL_HANDLE};
  XrSecureMrTensorPICO vstOutputRightUint8{XR_NULL_HANDLE};
  XrSecureMrTensorPICO vstOutputLeftFp32{XR_NULL_HANDLE};
  XrSecureMrTensorPICO vstOutputRightFp32{XR_NULL_HANDLE};
  XrSecureMrTensorPICO vstTimestamp{XR_NULL_HANDLE};
  XrSecureMrTensorPICO vstCameraMatrix{XR_NULL_HANDLE};
  XrSecureMrTensorPICO previousPosition{XR_NULL_HANDLE};
  XrSecureMrTensorPICO currentPosition{XR_NULL_HANDLE};
  XrSecureMrTensorPICO leftEyeUVGlobal{XR_NULL_HANDLE};
  XrSecureMrTensorPICO isFaceDetected{XR_NULL_HANDLE};

  XrSecureMrPipelineTensorPICO vstOutputLeftUint8Placeholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO vstOutputRightUint8Placeholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO vstOutputLeftFp32Placeholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO vstOutputRightFp32Placeholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO vstTimestampPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO vstCameraMatrixPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO previousRenderingPositionPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO currentRenderingPositionPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO previousPositionPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO currentPositionPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO uvPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO isFaceDetectedPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO visiblePlaceholder{XR_NULL_HANDLE};

  XrSecureMrPipelineTensorPICO vstImagePlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO timestampPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO cameraMatrixPlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO leftImgePlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO rightImagePlaceholder{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO leftEyeUVPlaceholder{XR_NULL_HANDLE};

  XrSecureMrTensorPICO gltfAsset{XR_NULL_HANDLE};
  XrSecureMrPipelineTensorPICO gltfPlaceholderTensor{XR_NULL_HANDLE};

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;
  bool keepRunning = true;
  bool pipelineAllInitialized = false;

  std::shared_ptr<SecureMR::Helper> m_apiHelper = SecureMR::CreateHelper(xr_instance, xr_session);
  ;
};

}  // namespace SecureMR
