# SecureMR Samples

Samples for SecureMR.

| Face tracking | Pose estimation | YOLO |
|:-------------:|:---------------:|:----:|
| ![Face tracking Demo](docs/Demo-UFO.gif) | ![Pose estimation demo](docs/Demo-Pose.gif) | ![YOLO demo](docs/Demo-YOLO.gif) |

## Project aims

The projects demonstrates the functionalities and usage of
the SecureMR interfaces through several out-of-the-box
sample applications. The applications each achieve some
customized MR-based effects with deployment of open-sourced
machine learning algorithms. 

Additionally, the project provides a set of utility classes,
located under `./base/securemr_utils` to simplify your
development of SecureMR-enabled applications. 

A docker file together with necessary resources are also 
contained under the `Docker/` directory, if you would like
deploy your own algorithm packages. 


## Repository Structure

```
.
├── Docker
|                Docker files and resources to convert ML algorithm packages
├── assets
│   │            Asset required by each sample project
│   │
│   ├── UFO
│   │            Assets used by sample "ufo" and "ufo_origin"
│   └── common
│                Assets shared by all sample projects
│
├── base
│   │            Base source codes, shared by sample projects,
│   │            including the fundermental OpenXR codes
│   │
│   ├── oxr_utils
│   │            Utility for fundermental OpenXR APIs, such as
│   │            verification XR API results and vulkan renderer
│   │
│   ├── securemr_utils
│   │            Utility for SecureMR samples, to simplify the logic
│   │            in samples. Note, to demonstrate the raw usage of
│   │            the C-API for SecureMR provided as an OpenXR extension,
│   │            some sample projects are written by directly calling
│   │            the C-API instead of using the utility classes here. 
│   │
│   └── vulkan_shaders
|                Vulkan shaders for the client
|
├── docs
│                Documentations
|
├── external
|                External dependencies
|
├── samples
│   │            Directory for all sample projects. 
│   │         
│   └── ufo
│               This is a sample showing a UFO "chasing" the human being
│               whoever it sees. The sample app uses an open-sourced
│               face detection model from MediaPipe.  
|
└── ...
```

## Prerequisite

#### (A) To run the demo, you will need

1. A PICO 4 Ultra device with the latest system update
1. Android Studio, with Android NDK installed, suggested NDK version = 25
1. Gradle and Android Gradle plugin (usually bundled with Android Studio install),
   suggested Gradle version = 8.7, Android Gradle Plugin version = 8.3.2
1. Java version at least 17 (required by the Android Gradle Plugin), recommended to be 21

#### (B) To run the docker, you will need

1. Docker desktop installed

## Deployment and Test

1. Install and configure according to the [prerequisite](#prerequisite). 
1. Open the repository root in Android Studio, as an Android project
1. After project sync, you will find there are four modules detected by the Android Studio, all under the `samples` folder: 
  1. `pose` which contains a pose detection demo
  1. `ufo` which contains a face detection demo
  1. `yolo` which contains an object detection demo
  1. `ufo-origin`, the same demo as `ufo`, but written using direct calls to the OpenXR C-API, with no 
      simplification using SecureMR Utils classes. 
1. Connect to a PICO 4 Ultra device with the latest OS update installed
1. Select the module you want to run, and click the launch button. 

