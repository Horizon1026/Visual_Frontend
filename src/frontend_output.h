#ifndef _VISUAL_FRONTEND_OUTPUT_H_
#define _VISUAL_FRONTEND_OUTPUT_H_

#include "datatype_basic.h"

namespace VISUAL_FRONTEND {

struct ObservePerView {
    uint32_t id = 0;    // Camera id. Used to differentiate like left and right camera.
    Vec2 raw_pixel_uv = Vec2::Zero();
    Vec2 rectified_norm_xy = Vec2::Zero();
};

using ObservePerFrame = std::vector<ObservePerView>;

struct FrontendOutputData {
    bool is_current_keyframe = true;
    std::vector<uint32_t> features_id;
    std::vector<ObservePerFrame> observes_per_frame;
};

}

#endif // end of _VISUAL_FRONTEND_OUTPUT_H_
