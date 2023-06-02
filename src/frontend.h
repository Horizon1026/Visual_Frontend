#ifndef _VISUAL_FRONTEND_H_
#define _VISUAL_FRONTEND_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"

#include "feature_point_detector.h"
#include "feature_harris.h"
#include "optical_flow.h"
#include "camera_basic.h"
#include "geometry_epipolar.h"
#include "image_processor.h"

#include "memory"

namespace VISUAL_FRONTEND {

struct FrontendOptions {
    uint32_t kImageRows = 0;
    uint32_t kImageCols = 0;
    uint32_t kMaxStoredFeaturePointsNumber = 120;
    bool kSelfSelectKeyframe = true;
    uint32_t kMinDetectedFeaturePointsNumberInCurrentImage = 60;
    float kMaxValidTrackBackPixelResidual = 1.0f;
};

class Frontend {

public:
	Frontend() = delete;
    Frontend(const uint32_t image_rows, const uint32_t image_cols);
    virtual ~Frontend();
    Frontend(const Frontend &frontend) = delete;

public:
    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const Image &image) { return false; }
    virtual bool RunOnce(const Image &image_left, const Image &image_right) { return false; }

    // Draw tracking results.
    virtual void DrawTrackingResults(const std::string title) {};

    // Reference for components.
    FrontendOptions &options() { return options_; }
    std::unique_ptr<FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature>> &feature_detector() { return feature_detector_; }
    std::unique_ptr<FEATURE_TRACKER::OpticalFlow> &feature_tracker() { return feature_tracker_; }
    std::unique_ptr<SENSOR_MODEL::CameraBasic> &camera_model() { return camera_model_; }
    std::unique_ptr<VISION_GEOMETRY::EpipolarSolver> &epipolar_solver() { return epipolar_solver_; }
    std::unique_ptr<IMAGE_PROCESSOR::ImageProcessor> &image_processor() { return image_processor_; }

    // Const eference for components.
    const FrontendOptions &options() const { return options_; }
    const std::unique_ptr<FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature>> &feature_detector() const { return feature_detector_; }
    const std::unique_ptr<FEATURE_TRACKER::OpticalFlow> &feature_tracker() const { return feature_tracker_; }
    const std::unique_ptr<SENSOR_MODEL::CameraBasic> &camera_model() const { return camera_model_; }
    const std::unique_ptr<VISION_GEOMETRY::EpipolarSolver> &epipolar_solver() const { return epipolar_solver_; }
    const std::unique_ptr<IMAGE_PROCESSOR::ImageProcessor> &image_processor() const { return image_processor_; }

    // Check every components.
    virtual bool CheckAllComponents();

public:
    // Options.
    FrontendOptions options_;

    // Components.
    std::unique_ptr<FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature>> feature_detector_ = nullptr;
    std::unique_ptr<FEATURE_TRACKER::OpticalFlow> feature_tracker_ = nullptr;
    std::unique_ptr<SENSOR_MODEL::CameraBasic> camera_model_ = nullptr;
    std::unique_ptr<VISION_GEOMETRY::EpipolarSolver> epipolar_solver_ = nullptr;
    std::unique_ptr<IMAGE_PROCESSOR::ImageProcessor> image_processor_ = nullptr;

    // Buffer allocated for visual frontend.
    uint8_t *stored_buff_ = nullptr;
    std::array<ImagePyramid, 4> stored_pyramids_;
    std::array<std::vector<uint32_t>, 2> stored_ids_ = {};
    std::array<std::vector<Vec2>, 8> stored_points_2d_ = {};
    std::array<std::vector<Vec2>, 2> stored_velocity_ = {};
    std::array<std::vector<Vec3>, 2> stored_points_3d_ = {};
    std::array<std::vector<uint32_t>, 2> stored_tracked_cnt_ = {};
    std::vector<uint8_t> tracked_status_;

    // Reference frame.
    ImagePyramid *ref_pyramid_left_ = &stored_pyramids_[0];
    ImagePyramid *ref_pyramid_right_ = &stored_pyramids_[1];
    std::vector<uint32_t> *ref_ids_ = &stored_ids_[0];
    std::vector<Vec2> *ref_pixel_uv_left_ = &stored_points_2d_[0];
    std::vector<Vec2> *ref_norm_xy_left_ = &stored_points_2d_[1];
    std::vector<Vec2> *ref_pixel_uv_right_ = &stored_points_2d_[2];
    std::vector<Vec2> *ref_norm_xy_right_ = &stored_points_2d_[3];
    std::vector<Vec3> *ref_points_xyz_ = &stored_points_3d_[0];
    std::vector<Vec2> *ref_vel_ = &stored_velocity_[0];
    std::vector<uint32_t> *ref_tracked_cnt_ = &stored_tracked_cnt_[0];

    // Current frame.
    ImagePyramid *cur_pyramid_left_ = &stored_pyramids_[2];
    ImagePyramid *cur_pyramid_right_ = &stored_pyramids_[3];
    std::vector<uint32_t> *cur_ids_ = &stored_ids_[1];
    std::vector<Vec2> *cur_pixel_uv_left_ = &stored_points_2d_[4];
    std::vector<Vec2> *cur_norm_xy_left_ = &stored_points_2d_[5];
    std::vector<Vec2> *cur_pixel_uv_right_ = &stored_points_2d_[6];
    std::vector<Vec2> *cur_norm_xy_right_ = &stored_points_2d_[7];
    std::vector<Vec3> *cur_points_xyz_ = &stored_points_3d_[1];
    std::vector<Vec2> *cur_vel_ = &stored_velocity_[1];
    std::vector<uint32_t> *cur_tracked_cnt_ = &stored_tracked_cnt_[1];

    // Feature index count;
    uint32_t feature_id_cnt_ = 1;

    // Keyframe flag.
    bool is_cur_image_keyframe_ = false;

};

}

#endif // end of _VISUAL_FRONTEND_H_
