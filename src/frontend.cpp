#include "frontend.h"

namespace VisualFrontend {

bool Frontend::RunOnce(const Image &cur_image) {
    if (cur_image.data() == nullptr) {
        return false;
    }

    if (ref_points_->size() != 0) {
        // Track features from ref pyramid to cur pyramid.
    }

    // Check if cur_pyarmid should be keyframe.

    return true;
}

}
