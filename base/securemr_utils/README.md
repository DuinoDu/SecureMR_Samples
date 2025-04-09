# Utility classes for SecureMR

SecureMR is available via the C-API of the OpenXR extension. 
However, the C-API might be diffcult to use and the repeated
codes can easily overflow the essential MR logics. Hence,
to simplify the major logics in our samples, and also to
ease the usage of SecureMR, we build these utility classes.

## Dependencies

The utility classes depend only on PICO's OpenXR library. 

## Key usage

NOTE: the utility classes are encapsulation of the 
OpenXR extension for PICO SecureMR. It does not 
change the behaviors of SecureMR. You may still
need to refer to the OpenXR extension's specification
for detailed usage and notices. 

### 1. Initialize a framework sesion

```cpp
auto frameworkSession =
    std::make_shared<SecureMr::FrameworkSession>(xr_instance, xr_session, 256, 256);
```


### 2. Create a SecureMR pipeline

```cpp
m_secureMrVSTImagePipeline = std::make_shared<SecureMr::Pipeline>(frameworkSession);
```

### 3. Define and use global tensors

```cpp
auto vstOutputLeftUint8Global = std::make_shared<SecureMr::GlobalTensor>(
                                frameworkSession,
                                SecureMr::TensorAttribute{
                                                            .dimensions = {256, 256},
                                                            .channels = 3,
                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO
                                                        }
                            );
                          
auto vstOutputRightUint8Global = std::make_shared<SecureMr::GlobalTensor>(
                                frameworkSession,
                                SecureMr::TensorAttribute{
                                                            .dimensions = {256, 256},
                                                            .channels = 3,
                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO
                                                        }
                            );

auto vstTimestampGlobal = std::make_shared<SecureMr::GlobalTensor>(
                                frameworkSession,
                                SecureMr::TensorAttribute{
                                                            .dimensions = {1},
                                                            .channels = 4,
                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO
                                                        }
                            );

auto vstCameraMatrixGlobal = std::make_shared<SecureMr::GlobalTensor>(
                                frameworkSession,
                                SecureMr::TensorAttribute{
                                                            .dimensions = {3, 3},
                                                            .channels = 1,
                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO
                                                        }
                            );
```

### 4. Create pipeline local tensors and placeholders

```cpp

auto leftEyeTransform = std::make_shared<SecureMr::PipelineTensor>(
                            m_secureMrVSTImagePipeline,
                            SecureMr::TensorAttribute{
                                                        .dimensions = {4, 4},
                                                        .channels = 1,
                                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO
                                                    }
                        );

auto vstOutputLeftUint8Placeholder = SecureMr::PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftUint8Global);
auto vstOutputRightUint8Placeholder = SecureMr::PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputRightUint8Global);
auto vstTimestampPlaceholder = SecureMr::PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstTimestampGlobal);
auto vstCameraMatrixPlaceholder = SecureMr::PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstCameraMatrixGlobal);
```

### 5. Build pipeline logic by adding operators to it

```cpp
(*m_secureMrVSTImagePipeline)
    .cameraAccess(vstOutputLeftUint8Placeholder, vstOutputRightUint8Placeholder,
                  vstTimestampPlaceholder, vstCameraMatrixPlaceholder)
    .camSpace2XrLocal(vstTimestampPlaceholder, nullptr, leftEyeTransform)
//...
```

### 6. Execute pipelines

```cpp
m_secureMrVSTImagePipeline->submit(
            {vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global},
            {vstOutputRightUint8Placeholder, vstOutputRightUint8Global},
            {vstTimestampPlaceholder, vstTimestampGlobal}
            {vstCameraMatrixPlaceholder, vstCameraMatrixGlobal},
            XR_NULL_HANDLE, nullptr);
```



