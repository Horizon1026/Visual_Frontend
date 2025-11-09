#ifndef _VISUAL_FRONTEND_OUTPUT_H_
#define _VISUAL_FRONTEND_OUTPUT_H_

#include "basic_type.h"

namespace VISUAL_FRONTEND {

/* Visual measurements of points. */
struct PointsObservePerView {
    Vec2 raw_pixel_uv = Vec2::Zero();
    Vec2 rectified_norm_xy = Vec2::Zero();
    Vec3 bearing_xyz = Vec3::Zero();
};
using PointsObservePerFrame = std::vector<PointsObservePerView>;
struct VisualPointsMeasure {
    bool is_current_keyframe = true;
    float time_stamp_s = -1.0f;
    int32_t direction_id = 0;
    std::vector<uint32_t> features_id;
    // observes_per_frame[feature_id][camera_id] -> PointsObservePerView.
    std::vector<PointsObservePerFrame> observes_per_frame;
};

/* Visual measurements of lines. */
struct LinesObservePerView {
    Vec4 raw_pixel_uv = Vec4::Zero();
    Vec4 rectified_norm_xy = Vec4::Zero();
};
using LinesObservePerFrame = std::vector<LinesObservePerView>;
struct VisualLinesMeasure {
    bool is_current_keyframe = true;
    float time_stamp_s = -1.0f;
    int32_t direction_id = 0;
    std::vector<uint32_t> features_id;
    // observes_per_frame[feature_id][camera_id] -> PointsObservePerView.
    std::vector<LinesObservePerFrame> observes_per_frame;
};

/* Visual measurements of all types. */
struct VisualMixMeasure {
    bool is_current_keyframe = true;
    float time_stamp_s = -1.0f;
    int32_t direction_id = 0;
    std::vector<uint32_t> points_id;
    std::vector<PointsObservePerFrame> points_observes_per_frame;
    std::vector<uint32_t> lines_id;
    std::vector<LinesObservePerFrame> lines_observes_per_frame;
};

}  // namespace VISUAL_FRONTEND

#endif  // end of _VISUAL_FRONTEND_OUTPUT_H_
