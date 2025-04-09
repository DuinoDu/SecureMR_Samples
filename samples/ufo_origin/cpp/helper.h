#pragma once
#include "pch.h"
#include <fstream>
#include <random>
#include <android/asset_manager.h>
#include "logger.h"
#include "common.h"

extern AAssetManager* g_assetManager;

namespace SecureMR {

#define GET_INTSANCE_PROC_ADDR(name, ptr) \
  CHECK_XRCMD(xrGetInstanceProcAddr(xr_instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&ptr)));

class Helper {
 public:
  XrSecureMrTensorPICO CreateTensorAsSlice(const XrSecureMrFrameworkPICO& framework, const std::vector<int>& start,
                                           const std::vector<int>& end, const std::vector<int>& skip, int32_t dimension,
                                           int32_t sliceSize);

  XrSecureMrPipelineTensorPICO CreatePipelineTensorAsSlice(const XrSecureMrPipelinePICO& pipeline,
                                                           const std::vector<int>& start, const std::vector<int>& end,
                                                           const std::vector<int>& skip, int32_t dimension,
                                                           int32_t sliceSize);

  void InitializePipelineIOPair(XrSecureMrPipelineIOPairPICO* pair, XrSecureMrPipelineTensorPICO placeholder,
                                XrSecureMrTensorPICO tensor);

  bool LoadModelData(const std::string& filePath, std::vector<char>& modelData);

  void CreateAndSetGlobalTensor(XrSecureMrFrameworkPICO framework, XrSecureMrTensorPICO& tensor,
                                const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                XrSecureMrTensorDataTypePICO tensorBitType, XrSecureMrTensorTypePICO tensorDataType,
                                const void* buffer, size_t bufferSize);

  void CreateAndSetPipelineTensor(XrSecureMrPipelinePICO pipeline, XrSecureMrPipelineTensorPICO& tensor,
                                  const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                                  XrSecureMrTensorDataTypePICO tensorBitType, XrSecureMrTensorTypePICO tensorDataType,
                                  const void* buffer, size_t bufferSize, bool isPlaceholder);

  void CreateGlobalTensor(XrSecureMrFrameworkPICO framework, XrSecureMrTensorPICO& tensor, const int32_t* dimensions,
                          uint32_t dimensionsCount, int8_t channel, XrSecureMrTensorDataTypePICO tensorBitType,
                          XrSecureMrTensorTypePICO tensorDataType, bool isPlaceholder);

  void CreatePipelineTensor(XrSecureMrPipelinePICO pipeline, XrSecureMrPipelineTensorPICO& tensor,
                            const int32_t* dimensions, uint32_t dimensionsCount, int8_t channel,
                            XrSecureMrTensorDataTypePICO tensorBitType, XrSecureMrTensorTypePICO tensorDataType,
                            bool isPlaceholder);

  //  void CreateOperatorWithConfig(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO op,
  //  XrSecureMrOperatorTypePICO operatorType, std::string config);

  void CreateOperator(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO& op,
                      XrSecureMrOperatorTypePICO operatorType);

  void CreateArithmeticOperator(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO& op, const char* config);

  void SetInput(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO op, XrSecureMrPipelineTensorPICO inputTensor,
                const char* inputName);

  void SetOutput(XrSecureMrPipelinePICO pipeline, XrSecureMrOperatorPICO op, XrSecureMrPipelineTensorPICO outputTensor,
                 const char* outputName);

  Helper(const XrInstance& instance, const XrSession& session) : xr_instance(instance), xr_session(session) {
    getInstanceProcAddr();
  }

  ~Helper() = default;

 protected:
  void getInstanceProcAddr() {
    Log::Write(Log::Level::Info, "getInstanceProcAddr start.");
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrOperatorPICO", xrCreateSecureMrOperatorPICO);
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrTensorPICO", xrCreateSecureMrTensorPICO);
    GET_INTSANCE_PROC_ADDR("xrCreateSecureMrPipelineTensorPICO", xrCreateSecureMrPipelineTensorPICO);
    GET_INTSANCE_PROC_ADDR("xrResetSecureMrTensorPICO", xrResetSecureMrTensorPICO);
    GET_INTSANCE_PROC_ADDR("xrResetSecureMrPipelineTensorPICO", xrResetSecureMrPipelineTensorPICO);
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorOperandByNamePICO", xrSetSecureMrOperatorOperandByNamePICO);
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorOperandByIndexPICO", xrSetSecureMrOperatorOperandByIndexPICO);
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorResultByNamePICO", xrSetSecureMrOperatorResultByNamePICO);
    GET_INTSANCE_PROC_ADDR("xrSetSecureMrOperatorResultByIndexPICO", xrSetSecureMrOperatorResultByIndexPICO);
    GET_INTSANCE_PROC_ADDR("xrExecuteSecureMrPipelinePICO", xrExecuteSecureMrPipelinePICO);
    Log::Write(Log::Level::Info, "getInstanceProcAddr end.");
  }

 private:
  XrInstance xr_instance;
  XrSession xr_session;

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
};

std::shared_ptr<Helper> CreateHelper(const XrInstance& instance, const XrSession& session);

}  // namespace SecureMR
