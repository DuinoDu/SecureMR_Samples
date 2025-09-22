// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "securemr_utils/serialization.h"

#include <fstream>
#include <system_error>
#include <stdexcept>
#include <unordered_map>

#include "oxr_utils/common.h"
#include "oxr_utils/logger.h"
#include "pipeline.h"
#include "tensor.h"

namespace SecureMR {

Json TensorAttributeToJson(const TensorAttribute& attr) {
  Json j;
  j["dimensions"] = attr.dimensions;
  j["channels"] = attr.channels;
  j["usage"] = static_cast<int>(attr.usage);
  j["data_type"] = static_cast<int>(attr.dataType);
  return j;
}

Json TensorAttributeVariantToJson(const std::variant<std::monostate, TensorAttribute>& attr) {
  Json j;
  if (std::holds_alternative<TensorAttribute>(attr)) {
    j = TensorAttributeToJson(std::get<TensorAttribute>(attr));
  } else {
    j["is_gltf"] = true;
  }
  return j;
}

Json TensorListToJson(const std::vector<std::string>& tensors) {
  Json arr = Json::array();
  for (const auto& name : tensors) {
    arr.push_back(name);
  }
  return arr;
}

Json MappedTensorListToJson(const std::vector<std::pair<std::string, std::string>>& mapping) {
  Json arr = Json::array();
  for (const auto& [alias, tensor] : mapping) {
    Json entry;
    entry["name"] = alias;
    entry["tensor"] = tensor;
    arr.push_back(entry);
  }
  return arr;
}

bool WriteJsonToFile(const std::filesystem::path& filePath, const Json& spec) {
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "WriteJsonToFile failed: writable path unavailable");
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(filePath.parent_path(), ec);
  std::ofstream ofs(filePath);
  if (!ofs) {
    Log::Write(Log::Level::Error,
               Fmt("WriteJsonToFile failed: cannot open %s", filePath.string().c_str()));
    return false;
  }
  ofs << spec.dump(2);
  return true;
}

bool JsonToTensorAttribute(const Json& j, TensorAttribute& out) {
  if (j.find("dimensions") == j.end() || j.find("channels") == j.end() || j.find("usage") == j.end() ||
      j.find("data_type") == j.end()) {
    return false;
  }
  if (!j["dimensions"].is_array()) {
    return false;
  }
  out.dimensions.clear();
  for (const auto& dim : j["dimensions"]) {
    if (!dim.is_number_integer()) {
      return false;
    }
    out.dimensions.push_back(dim.get<int>());
  }
  out.channels = static_cast<int8_t>(j["channels"].get<int>());
  out.usage = static_cast<XrSecureMrTensorTypePICO>(j["usage"].get<int>());
  out.dataType = static_cast<XrSecureMrTensorDataTypePICO>(j["data_type"].get<int>());
  return true;
}

std::vector<std::string> ParseTensorList(const Json& arr) {
  std::vector<std::string> tensors;
  if (!arr.is_array()) {
    return tensors;
  }
  tensors.reserve(arr.size());
  for (const auto& each : arr) {
    if (each.is_string()) {
      tensors.push_back(each.get<std::string>());
    } else if (each.is_object()) {
      if (auto tensorIt = each.find("tensor"); tensorIt != each.end() && tensorIt->is_string()) {
        tensors.push_back(tensorIt->get<std::string>());
      }
    }
  }
  return tensors;
}

std::vector<std::pair<std::string, std::string>> ParseMappedTensorList(const Json& arr) {
  std::vector<std::pair<std::string, std::string>> mapping;
  if (!arr.is_array()) {
    return mapping;
  }
  mapping.reserve(arr.size());
  for (const auto& each : arr) {
    std::string tensorName;
    std::string alias;
    if (each.is_object()) {
      if (auto aliasIt = each.find("name"); aliasIt != each.end() && aliasIt->is_string()) {
        alias = aliasIt->get<std::string>();
      }
      if (auto tensorIt = each.find("tensor"); tensorIt != each.end() && tensorIt->is_string()) {
        tensorName = tensorIt->get<std::string>();
      }
    } else if (each.is_string()) {
      tensorName = each.get<std::string>();
      alias = tensorName;
    }
    if (!tensorName.empty()) {
      if (alias.empty()) {
        alias = tensorName;
      }
      mapping.emplace_back(alias, tensorName);
    }
  }
  return mapping;
}

bool JsonToFloatArray(const Json& arr, std::array<float, 6>& dest) {
  if (!arr.is_array() || arr.size() != dest.size()) {
    return false;
  }
  for (size_t i = 0; i < dest.size(); ++i) {
    if (!arr[i].is_number()) {
      return false;
    }
    dest[i] = arr[i].get<float>();
  }
  return true;
}

Json LoadJsonFromFile(const std::filesystem::path& filePath) {
  Json parsed;
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "LoadJsonFromFile failed: path empty");
    return parsed;
  }
  std::ifstream ifs(filePath);
  if (!ifs) {
    Log::Write(Log::Level::Error,
               Fmt("LoadJsonFromFile failed: cannot open %s", filePath.string().c_str()));
    return parsed;
  }
  try {
    ifs >> parsed;
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("LoadJsonFromFile failed: %s", e.what()));
    parsed = Json();
  }
  return parsed;
}

bool DeserializePipelineFromJson(const Json& spec,
                                 const std::shared_ptr<FrameworkSession>& session,
                                 PipelineDeserializationResult& outResult,
                                 std::string& outError,
                                 const PipelineDeserializationOptions& options) {
  outResult = {};
  outError.clear();
  if (!spec.is_object()) {
    outError = "JSON is not an object";
    return false;
  }

  const auto tensorsIt = spec.find("tensors");
  if (tensorsIt == spec.end() || !tensorsIt->is_object()) {
    outError = "tensors section missing or invalid";
    return false;
  }

  auto pipeline = std::make_shared<Pipeline>(session);
  for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
    const std::string tensorName = it.key();
    const Json& tensorSpec = *it;
    const bool isPlaceholder = tensorSpec.value("is_placeholder", false);
    const bool isGltf = tensorSpec.value("is_gltf", false);
    std::shared_ptr<PipelineTensor> tensor;
    if (isPlaceholder && isGltf) {
      tensor = PipelineTensor::PipelineGLTFPlaceholder(pipeline);
    } else {
      TensorAttribute attr{};
      if (!JsonToTensorAttribute(tensorSpec, attr)) {
        outError = Fmt("invalid tensor attribute for %s", tensorName.c_str());
        return false;
      }
      tensor = std::make_shared<PipelineTensor>(pipeline, attr, isPlaceholder);
    }
    outResult.tensorMap.emplace(tensorName, std::move(tensor));
  }

  const auto requireTensor = [&](const std::string& name) -> std::shared_ptr<PipelineTensor> {
    auto it = outResult.tensorMap.find(name);
    if (it == outResult.tensorMap.end()) {
      throw std::runtime_error(Fmt("tensor '%s' not found", name.c_str()));
    }
    return it->second;
  };

  const auto operatorsIt = spec.find("operators");
  if (operatorsIt == spec.end() || !operatorsIt->is_array()) {
    outError = "operators section missing or invalid";
    return false;
  }

  try {
    for (const auto& opSpec : *operatorsIt) {
      const std::string type = opSpec.value("type", "");
      const auto inputs = ParseTensorList(opSpec.value("inputs", Json::array()));
      const auto outputs = ParseTensorList(opSpec.value("outputs", Json::array()));

      auto requireByIndex = [&](const std::vector<std::string>& container, size_t index,
                                const char* what) -> std::shared_ptr<PipelineTensor> {
        if (index >= container.size()) {
          throw std::runtime_error(Fmt("%s index %zu out of range", what, index));
        }
        return requireTensor(container[index]);
      };

      if (type == "camera_access") {
        if (outputs.size() != 4) {
          throw std::runtime_error("camera_access outputs malformed");
        }
        pipeline->cameraAccess(requireByIndex(outputs, 0, "camera_access output"),
                               requireByIndex(outputs, 1, "camera_access output"),
                               requireByIndex(outputs, 2, "camera_access output"),
                               requireByIndex(outputs, 3, "camera_access output"));
      } else if (type == "get_affine") {
        std::array<float, 6> src{};
        std::array<float, 6> dst{};
        if (!JsonToFloatArray(opSpec["src_points"], src) || !JsonToFloatArray(opSpec["dst_points"], dst)) {
          throw std::runtime_error("get_affine points malformed");
        }
        if (outputs.empty()) {
          throw std::runtime_error("get_affine requires output tensor");
        }
        pipeline->getAffine(src, dst, requireTensor(outputs.front()));
      } else if (type == "apply_affine") {
        if (inputs.size() < 2 || outputs.empty()) {
          throw std::runtime_error("apply_affine requires two inputs and one output");
        }
        pipeline->applyAffine(requireByIndex(inputs, 0, "apply_affine input"),
                              requireByIndex(inputs, 1, "apply_affine input"),
                              requireByIndex(outputs, 0, "apply_affine output"));
      } else if (type == "assignment") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("assignment requires input and output tensors");
        }
        pipeline->assignment(requireByIndex(inputs, 0, "assignment input"),
                             requireByIndex(outputs, 0, "assignment output"));
      } else if (type == "cvt_color") {
        const int flag = opSpec.value("flag", 0);
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("cvt_color requires input and output tensors");
        }
        pipeline->cvtColor(flag, requireByIndex(inputs, 0, "cvt_color input"),
                           requireByIndex(outputs, 0, "cvt_color output"));
      } else if (type == "type_convert") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("type_convert requires input and output tensors");
        }
        pipeline->typeConvert(requireByIndex(inputs, 0, "type_convert input"),
                              requireByIndex(outputs, 0, "type_convert output"));
      } else if (type == "arithmetic") {
        const std::string expression = opSpec.value("expression", "");
        std::vector<std::shared_ptr<PipelineTensor>> operands;
        operands.reserve(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
          operands.push_back(requireByIndex(inputs, idx, "arithmetic input"));
        }
        if (outputs.empty()) {
          throw std::runtime_error("arithmetic requires output tensor");
        }
        pipeline->arithmetic(expression, operands, requireByIndex(outputs, 0, "arithmetic output"));
      } else {
        bool handled = false;
        if (options.customOperatorHandler) {
          handled = options.customOperatorHandler(opSpec, requireTensor, pipeline, outError);
        }
        if (!handled) {
          throw std::runtime_error(Fmt("unsupported operator type '%s'", type.c_str()));
        }
      }
    }
  } catch (const std::exception& e) {
    if (outError.empty()) {
      outError = e.what();
    }
    return false;
  }

  outResult.pipeline = std::move(pipeline);
  return true;
}

}  // namespace SecureMR
