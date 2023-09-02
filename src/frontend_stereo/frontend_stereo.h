#ifndef _VISUAL_FRONTEND_STEREO_H_
#define _VISUAL_FRONTEND_STEREO_H_

#include "frontend.h"

#include "descriptor_brief.h"
#include "descriptor_matcher.h"

namespace VISUAL_FRONTEND {

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
    virtual void DrawTrackingResults(const std::string title) override;
    // Support for log recording.
    virtual void RegisterLogPackages() override {}

private:
    bool ProcessSourceImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right);

private:
    std::unique_ptr<FEATURE_DETECTOR::BriefDescriptor> descriptor_ = std::make_unique<FEATURE_DETECTOR::BriefDescriptor>();
    std::unique_ptr<BriefMatcher> feature_matcher_ = std::make_unique<BriefMatcher>();

    std::vector<Vec2> detected_features_in_cur_right_;
    std::vector<FEATURE_DETECTOR::BriefType> cur_descriptor_left_;
    std::vector<FEATURE_DETECTOR::BriefType> cur_descriptor_right_;
    std::vector<Vec2> predicted_pixel_uv_in_cur_right_;
};

}

#endif // end of _VISUAL_FRONTEND_STEREO_H_
