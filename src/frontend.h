#ifndef _VISUAL_FRONTEND_H_
#define _VISUAL_FRONTEND_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"

namespace VisualFrontend {

struct FrontendOptions {
    uint32_t image_rows = 0;
    uint32_t image_cols = 0;
};

class Frontend {

public:
	Frontend() = delete;
    Frontend(const uint32_t image_rows, const uint32_t image_cols);
    virtual ~Frontend();
    Frontend(const Frontend &frontend) = delete;

public:
    // Frontend is driven by mono image image or stereo images.
    bool RunOnce(const Image &image);
    bool RunOnce(const Image &image_left, const Image &image_right);

    FrontendOptions &options() { return options_; }

private:
    // Options.
    FrontendOptions options_;

    // Buffer allocated for visual frontend.
    uint8_t *stored_buff_ = nullptr;
    ImagePyramid stored_pyramids_[4];
    std::vector<uint32_t> stored_ids_[2] = {};
    std::vector<Vec2> stored_points_[2] = {};

    // Reference frame.
    ImagePyramid *ref_pyramid_left_ = &stored_pyramids_[0];
    ImagePyramid *ref_pyramid_right_ = &stored_pyramids_[1];
    std::vector<uint32_t> *ref_ids_ = &stored_ids_[0];
    std::vector<Vec2> *ref_points_ = &stored_points_[0];

    // Current frame.
    ImagePyramid *cur_pyramid_left_ = &stored_pyramids_[2];
    ImagePyramid *cur_pyramid_right_ = &stored_pyramids_[3];
    std::vector<uint32_t> *cur_ids_ = &stored_ids_[1];
    std::vector<Vec2> *cur_points_ = &stored_points_[1];
};

}

#endif
