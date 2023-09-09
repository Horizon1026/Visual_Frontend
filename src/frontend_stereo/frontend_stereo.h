#ifndef _VISUAL_FRONTEND_STEREO_H_
#define _VISUAL_FRONTEND_STEREO_H_

#include "frontend.h"

#include "descriptor_brief.h"
#include "descriptor_matcher.h"

namespace VISUAL_FRONTEND {

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
};

#pragma pack()

/* Class BriefMatcher Declaration. */
class BriefMatcher : public FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::BriefType> {

public:
    BriefMatcher() : FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::BriefType>() {}
    virtual ~BriefMatcher() = default;

    virtual int32_t ComputeDistance(const FEATURE_DETECTOR::BriefType &descriptor_ref,
                                    const FEATURE_DETECTOR::BriefType &descriptor_cur) override {
        if (descriptor_ref.empty() || descriptor_cur.empty()) {
            return kMaxInt32;
        }

        int32_t distance = 0;
        for (uint32_t i = 0; i < descriptor_ref.size(); ++i) {
            if (descriptor_ref[i] != descriptor_cur[i]) {
                ++distance;
            }
        }
        return distance;
    }
};

/* Class FrontendStereo Declaration. */
class FrontendStereo : public Frontend {

public:
	FrontendStereo() = delete;
    FrontendStereo(const uint32_t image_rows, const uint32_t image_cols) : Frontend(image_rows, image_cols) {}
    virtual ~FrontendStereo() = default;
    FrontendStereo(const FrontendStereo &frontend_stereo) = delete;

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const GrayImage &image_left, const GrayImage &image_right) override;
    // Draw tracking results.
    virtual void DrawTrackingResults(const std::string &title) override;
    // Support for log recording.
    virtual void RegisterLogPackages() override;
    // Update frontend result.
    virtual void UpdateFrontendOutputData() override;

    // Reference for member variables.
    std::unique_ptr<FEATURE_DETECTOR::BriefDescriptor> &descriptor() { return descriptor_; }
    std::unique_ptr<BriefMatcher> &feature_matcher() { return feature_matcher_; }
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
    std::unique_ptr<FEATURE_DETECTOR::BriefDescriptor> descriptor_ = nullptr;
    std::unique_ptr<BriefMatcher> feature_matcher_ = nullptr;

    // Support for stereo tracking.
    Pixel half_patch_size_for_stereo_tracking_ = Pixel(2, 25);
    std::array<std::vector<uint8_t>, 2> stereo_tracked_status_;
    std::vector<uint8_t> *ref_stereo_tracked_status_ = &stereo_tracked_status_[0];
    std::vector<uint8_t> *cur_stereo_tracked_status_ = &stereo_tracked_status_[1];

    // Support for stereo matching.
    std::vector<Vec2> detected_features_in_cur_right_;
    std::vector<FEATURE_DETECTOR::BriefType> cur_descriptor_left_;
    std::vector<FEATURE_DETECTOR::BriefType> cur_descriptor_right_;
    std::vector<Vec2> predicted_pixel_uv_in_cur_right_;

    // Temp vector for tracking back.
    std::vector<Vec2> ref_pixel_uv_left_tracked_back_;
    // Temp package data for log file.
    FrontendStereoLog log_package_data_;
};

}

#endif // end of _VISUAL_FRONTEND_STEREO_H_
