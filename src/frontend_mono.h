#ifndef _VISUAL_FRONTEND_MONO_H_
#define _VISUAL_FRONTEND_MONO_H_

#include "frontend.h"

namespace VISUAL_FRONTEND {

class FrontendMono : public Frontend {

public:
	FrontendMono() = delete;
    FrontendMono(const uint32_t image_rows, const uint32_t image_cols) : Frontend(image_rows, image_cols) {}
    virtual ~FrontendMono() = default;
    FrontendMono(const FrontendMono &frontend_mono) = delete;

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const GrayImage &image) override;

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
    std::vector<Vec2> ref_pixel_xy_left_tracked_back_;
};

}

#endif // end of _VISUAL_FRONTEND_MONO_H_
