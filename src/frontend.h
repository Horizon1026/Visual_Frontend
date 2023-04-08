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

};

}

#endif
