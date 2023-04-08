#ifndef _VISUAL_FRONTEND_H_
#define _VISUAL_FRONTEND_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"

namespace VisualFrontend {

class Frontend {

public:
	Frontend() = default;
    virtual ~Frontend() = default;
    Frontend(const Frontend &frontend) = delete;

public:
    // Frontend is driven by mono image pyramid or stereo image pyramids.
    bool RunOnce(const ImagePyramid &pyramid);
    bool RunOnce(const ImagePyramid &pyramid_left, const ImagePyramid &pyramid_right);

};

}

#endif
