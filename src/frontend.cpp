#include "frontend.h"
#include "slam_memory.h"

namespace VisualFrontend {

Frontend::Frontend(const uint32_t image_rows, const uint32_t image_cols) {
    const uint32_t size = image_rows * image_cols;
    stored_buff_ = (uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * size * 8);

    // Allocate memory for stored buffer.
    uint8_t *buf = stored_buff_;
    for (uint32_t i = 0; i < 4; ++i) {
        stored_pyramids_[i].SetRawImage(buf, image_rows, image_cols);
        buf += size;
        stored_pyramids_[i].SetPyramidBuff(buf);
        buf += size;
    }

    // Config feature detector.
    feature_detector_.options().kMethod = FEATURE_DETECTOR::FeatureDetector::HARRIS;
    feature_detector_.options().kMinValidResponse = 20.0f;
    feature_detector_.options().kMinFeatureDistance = 20;

    // Config optical flow tracker.
    feature_tracker_.options().kPatchRowHalfSize = 6;
    feature_tracker_.options().kPatchColHalfSize = 6;
    feature_tracker_.options().kMethod = OPTICAL_FLOW::LkMethod::LK_FAST;

}

Frontend::~Frontend() {
    SlamMemory::Free(stored_buff_);
}

bool Frontend::RunOnce(const Image &cur_image) {
    if (cur_image.data() == nullptr) {
        return false;
    }

    if (ref_points_->size() != 0) {
        // Track features from ref pyramid to cur pyramid.
    }

    // Check if cur_pyarmid should be keyframe.

    return true;
}

}
