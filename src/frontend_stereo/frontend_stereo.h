#ifndef _VISUAL_FRONTEND_STEREO_H_
#define _VISUAL_FRONTEND_STEREO_H_

#include "frontend.h"

namespace visual_frontend {

/* Packages of log to be recorded. */
#pragma pack(1)

struct FrontendStereoLog {
    uint8_t is_keyframe = 0;
    uint32_t num_of_old_features_in_only_left = 0;
    uint32_t num_of_tracked_features_in_only_left = 0;
    uint32_t num_of_inliers_in_only_left = 0;
    uint32_t num_of_inliers_after_filter_in_only_left = 0;
    uint32_t num_of_tracked_feature_from_left_to_right = 0;
    uint32_t num_of_inliers_from_left_to_right = 0;
    uint32_t num_of_new_features_in_only_left = 0;
    float cost_time_ms_of_loop = 0.0f;
};

#pragma pack()

/* Class FrontendStereo Declaration. */
class FrontendStereo: public Frontend {

public:
    FrontendStereo() = delete;
    FrontendStereo(const uint32_t image_rows, const uint32_t image_cols): Frontend(image_rows, image_cols) {}
    virtual ~FrontendStereo() = default;

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const GrayImage &image_left, const GrayImage &image_right, const float time_stamp_s) override;
    // Draw tracking results.
    virtual void DrawTrackingResults(const std::string &title, const float time_stamp_s) override;
    // Support for log recording.
    virtual void RegisterLogPackages() override;
    // Update frontend result.
    virtual void UpdateFrontendOutputData(const float time_stamp_s) override;

    // Reference for member variables.
    Pixel &half_patch_size_for_stereo_tracking() { return half_patch_size_for_stereo_tracking_; }

private:
    bool ProcessSourceImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right);
    bool PredictPixelLocationInCurrentLeftImage();
    bool TrackFeaturesFromRefernceLeftToCurrentLeftImage();
    bool RejectOutliersBetweenRefernceLeftAndCurrentLeftImage();
    bool ComputeOpticalFlowVelocity();
    bool SparsifyTrackedFeaturesInLeft();
    bool TrackFeaturesFromCurrentLeftToCurrentRightImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right);
    bool RejectOutliersBetweenCurrentLeftToCurrentRightImage();
    bool SelectKeyframe();
    bool AdjustTrackingResultByStatus();
    bool SupplememtNewFeatures(const GrayImage &cur_image_left);
    bool MakeCurrentFrameKeyframe();

private:
    // Support for stereo tracking.
    Pixel half_patch_size_for_stereo_tracking_ = Pixel(2, 25);
    std::array<std::vector<uint8_t>, 2> stereo_tracked_status_;
    std::vector<uint8_t> *ref_stereo_tracked_status_ = &stereo_tracked_status_[0];
    std::vector<uint8_t> *cur_stereo_tracked_status_ = &stereo_tracked_status_[1];
    // Temp vector for tracking back.
    std::vector<Vec2> ref_pixel_uv_left_tracked_back_;
    // Temp package data for log file.
    FrontendStereoLog log_package_data_;
};

}  // namespace visual_frontend

#endif  // end of _VISUAL_FRONTEND_STEREO_H_
