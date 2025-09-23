plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    compileSdk = 32
    ndkVersion = "26.3.11579264"
    namespace = "com.bytedance.pico.secure_mr_demo.mnistwild"
    defaultConfig {
        minSdk = 32
        targetSdk = 32
        versionCode = 1
        versionName = "1.0"
        applicationId = "com.bytedance.pico.secure_mr_demo.mnistwild"
        externalNativeBuild {
            cmake {
                arguments.add("-DANDROID_STL=c++_shared")
                arguments.add("-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF")
                // Toggle JSON-based pipeline via: -PmnistwildUseAssetJson=true
                val useAssetJson = (project.findProperty("mnistwildUseAssetJson") as String?)?.toBoolean() ?: false
                if (useAssetJson) {
                    arguments.add("-DMNISTWILD_USE_ASSET_JSON=ON")
                }
            }
            ndk {
                abiFilters.add("arm64-v8a")
            }
        }
    }
    lint {
        disable.add("ExpiredTargetSdkVersion")
    }
    buildFeatures {
        prefab = true
    }
    buildTypes {
        getByName("debug") {
            isDebuggable = true
            isJniDebuggable = true
        }
        getByName("release") {
            isDebuggable = false
            isJniDebuggable = false
        }
    }
    externalNativeBuild {
        cmake {
            version = "3.22.1"
            path("CMakeLists.txt")
        }
    }
    sourceSets {
        getByName("main") {
            manifest.srcFile("AndroidManifest.xml")
            assets.srcDirs("../../assets/common", "../../assets/mnistwild")
        }
    }
    packaging {
        jniLibs {
            keepDebugSymbols.add("**.so")
        }
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}
