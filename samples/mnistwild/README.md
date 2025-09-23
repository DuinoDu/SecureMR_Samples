# MNIST Demo (SecureMR Utils)

Refactored MNIST Wild sample implemented with `securemr_utils`. See `cpp/` sources for details.

## Build
There are two types build:

1. Build pipeline in cpp and serializing pipeline to json file.
```
./gradlew samples:mnistwild
```

2. Load pipeline from json file only.

```
./gradlew samples:mnistwild -PmnistwildUseAssetJson=true
```
