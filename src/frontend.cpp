#include "frontend.h"
#include "slam_memory.h"

// Debug.
#include "opencv2/opencv.hpp"

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

    // Image process.
    std::copy_n(cur_image.data(), cur_image.rows() * cur_image.cols(), cur_pyramid_left_->GetImage(0).data());
    cur_pyramid_left_->CreateImagePyramid(4);

    for (uint32_t i = 0; i < cur_pyramid_left_->level(); ++i) {
        Image one_level = cur_pyramid_left_->GetImage(i);
        cv::Mat image(one_level.rows(), one_level.cols(), CV_8UC1, one_level.data());
        cv::imshow(std::to_string(i), image);
        cv::waitKey(1);
    }
    cv::waitKey(0);

    if (ref_points_->size() != 0) {
        // Track features from ref pyramid to cur pyramid.
        cur_points_->clear();
        cur_ids_->clear();
        cur_status_->clear();

        feature_tracker_.TrackMultipleLevel(*ref_pyramid_left_, *cur_pyramid_left_, *ref_points_, *cur_points_, *cur_status_);

    }

    // Check if cur_pyarmid should be keyframe.

    return true;
}

}
