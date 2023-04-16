#ifndef _VISUAL_FRONTEND_STEREO_H_
#define _VISUAL_FRONTEND_STEREO_H_

#include "frontend.h"

namespace VISUAL_FRONTEND {

class FrontendStereo : public Frontend {

public:
	FrontendStereo() = delete;
    FrontendStereo(const uint32_t image_rows, const uint32_t image_cols) : Frontend(image_rows, image_cols) {}
    virtual ~FrontendStereo() = default;
    FrontendStereo(const FrontendStereo &frontend_stereo) = delete;

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const Image &image_left, const Image &image_right) override;

    // Visualize API.
    FrontendStereoVisualizeFunctionPtr VisualizeResult = nullptr;

private:
    bool ProcessSourceImage(const Image &cur_image_left, const Image &cur_image_right);
};

}

#endif // end of _VISUAL_FRONTEND_STEREO_H_
