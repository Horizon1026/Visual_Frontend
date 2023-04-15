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

public:
    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const Image &image) override;
};

}

#endif // end of _VISUAL_FRONTEND_MONO_H_
