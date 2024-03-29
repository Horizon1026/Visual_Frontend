#ifndef _VISUAL_FRONTEND_MONO_H_
#define _VISUAL_FRONTEND_MONO_H_

#include "frontend.h"

namespace VISUAL_FRONTEND {

/* Packages of log to be recorded. */
#pragma pack(1)

struct FrontendMonoLog {
    uint8_t is_keyframe = 0;
    uint32_t num_of_old_features = 0;
    uint32_t num_of_tracked_features = 0;
    uint32_t num_of_inliers = 0;
    uint32_t num_of_inliers_after_filter = 0;
    uint32_t num_of_new_features = 0;
    float cost_time_ms_of_loop = 0.0f;
};

#pragma pack()

/* Class FrontendMono Declaration. */
class FrontendMono : public Frontend {

public:
	FrontendMono() = delete;
    FrontendMono(const uint32_t image_rows, const uint32_t image_cols) : Frontend(image_rows, image_cols) {}
    virtual ~FrontendMono() = default;

    // Frontend is driven by mono image.
    virtual bool RunOnce(const GrayImage &image, const float time_stamp_s) override;
    // Draw tracking results.
    virtual void DrawTrackingResults(const std::string &title, const float time_stamp_s) override;
    // Support for log recording.
    virtual void RegisterLogPackages() override;
    // Update frontend result.
    virtual void UpdateFrontendOutputData(const float time_stamp_s) override;

private:
    bool ProcessSourceImage(const GrayImage &cur_image);
    bool PredictPixelLocation();
    bool TrackFeatures();
    bool ComputeOpticalFlowVelocity();
    bool LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem();
    bool RejectOutliersByEpipolarConstrain();
    bool RejectOutliersByTrackingBack();
    bool SparsifyTrackedFeatures();
    bool SelectKeyframe();
    bool AdjustTrackingResultByStatus();
    bool SupplementNewFeatures(const GrayImage &cur_image_left);
    bool MakeCurrentFrameKeyframe();

private:
    // Temp vector for tracking back.
    std::vector<Vec2> ref_pixel_uv_left_tracked_back_;
    // Temp package data for log file.
    FrontendMonoLog log_package_data_;
};

}

#endif // end of _VISUAL_FRONTEND_MONO_H_
