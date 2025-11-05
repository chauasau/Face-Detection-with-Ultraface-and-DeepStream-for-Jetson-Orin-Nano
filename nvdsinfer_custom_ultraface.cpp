#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

extern "C"
bool NvDsInferParseCustomUltraface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    // Find output layers
    const NvDsInferLayerInfo *scoreLayer = nullptr;
    const NvDsInferLayerInfo *bboxLayer = nullptr;
    for (auto &layer : outputLayersInfo) {
        if (strcmp(layer.layerName, "scores") == 0)
            scoreLayer = &layer;
        else if (strcmp(layer.layerName, "boxes") == 0)
            bboxLayer = &layer;
    }

    if (!scoreLayer || !bboxLayer) {
        std::cerr << "Ultraface parser: missing output layers (scores / boxes)" << std::endl;
        return false;
    }

    const float *scores = (const float *)scoreLayer->buffer;
    const float *boxes = (const float *)bboxLayer->buffer;
    const int numAnchors = bboxLayer->inferDims.d[0];
    const int numClasses = scoreLayer->inferDims.d[1];

    const float threshold = detectionParams.perClassThreshold[0];

    for (int i = 0; i < numAnchors; ++i) {
        float conf = scores[i * numClasses + 1]; // face class confidence
        if (conf < threshold)
            continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = 0; // face
        obj.detectionConfidence = conf;
        obj.left = boxes[i * 4 + 0] * networkInfo.width;
        obj.top = boxes[i * 4 + 1] * networkInfo.height;
        obj.width = (boxes[i * 4 + 2] - boxes[i * 4 + 0]) * networkInfo.width;
        obj.height = (boxes[i * 4 + 3] - boxes[i * 4 + 1]) * networkInfo.height;

        if (obj.width <= 0 || obj.height <= 0)
            continue;

        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomUltraface);
