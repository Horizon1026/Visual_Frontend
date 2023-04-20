#ifndef _VISUAL_FRONTEND_MONO_H_
#define _VISUAL_FRONTEND_MONO_H_

#include "frontend.h"

namespace VISUAL_FRONTEND {

// Visualize api for mono frontend.
typedef void (*FrontendMonoVisualizeFunctionPtr)(const std::string,                 // title.
                                                 const Image &, const Image &,      // ref_image, cur_image.
                                                 const std::vector<Vec2> &,         // ref_pixel_uv.
                                                 const std::vector<Vec2> &,         // cur_pixel_uv.
                                                 const std::vector<uint32_t> &,     // ref_ids.
                                                 const std::vector<uint32_t> &,     // cur_ids.
                                                 const std::vector<uint32_t> &,     // ref_features_tracked_times.
                                                 const std::vector<Vec2> &);        // ref_features_optical_flow_velocity.

class FrontendMono : public Frontend {

public:
	FrontendMono() = delete;
    FrontendMono(const uint32_t image_rows, const uint32_t image_cols) : Frontend(image_rows, image_cols) {}
    virtual ~FrontendMono() = default;
    FrontendMono(const FrontendMono &frontend_mono) = delete;

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const Image &image) override;

    // Visualize API.
    FrontendMonoVisualizeFunctionPtr VisualizeResult = nullptr;

private:
    bool ProcessSourceImage(const Image &cur_image);
    bool PredictPixelLocation();
    bool TrackFeatures();
    bool ComputeOpticalFlowVelocity();
    bool LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem();
    bool RejectOutliersByEpipolarConstrain();
    bool RejectOutliersByTrackingBack();
    bool SparsifyTrackedFeatures();
    bool SelectKeyframe();
    bool AdjustTrackingResultByStatus();
    bool SupplementNewFeatures(const Image &cur_image_left);
    bool MakeCurrentFrameKeyframe();

private:
    // Temp vector for tracking back.
    std::vector<Vec2> ref_pixel_xy_left_tracked_back_ = {};
};

}

#endif // end of _VISUAL_FRONTEND_MONO_H_
