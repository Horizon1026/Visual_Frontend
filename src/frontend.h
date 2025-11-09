#ifndef _VISUAL_FRONTEND_H_
#define _VISUAL_FRONTEND_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"

#include "frontend_output.h"

#include "feature_fast.h"
#include "feature_harris.h"
#include "feature_point_detector.h"

#include "camera_basic.h"
#include "geometry_epipolar.h"
#include "image_processor.h"
#include "optical_flow.h"

#include "binary_data_log.h"

#include "memory"

namespace visual_frontend {

/* Options for Class Frontend. */
struct FrontendOptions {
    uint32_t kImageRows = 0;
    uint32_t kImageCols = 0;
    uint32_t kMaxStoredFeaturePointsNumber = 120;
    bool kSelfSelectKeyframe = true;
    uint32_t kMinDetectedFeaturePointsNumberInCurrentImage = 60;
    uint32_t kImagePyramidLevels = 4;
    float kMaxValidTrackBackPixelResidual = 1.0f;
    bool kEnableRecordBinaryCurveLog = false;
    bool kEnableRecordBinaryImageLog = false;
    bool kEnableShowVisualizeResult = true;
};

/* Class Frontend Declaration. */
class Frontend {

public:
    Frontend() = delete;
    Frontend(const uint32_t image_rows, const uint32_t image_cols);
    virtual ~Frontend();
    Frontend(const Frontend &frontend) = delete;

    // Perpare for log recording.
    bool Initialize(const std::string &log_file_name = "./frontend_log.binlog");

    // Frontend is driven by mono image or stereo images.
    virtual bool RunOnce(const GrayImage &image, const float time_stamp_s) { return false; }
    virtual bool RunOnce(const GrayImage &image_left, const GrayImage &image_right, const float time_stamp_s) { return false; }
    // Draw tracking results.
    virtual void DrawTrackingResults(const std::string &title, const float time_stamp_s) = 0;
    // Support for log recording.
    virtual void RegisterLogPackages() = 0;
    // Update frontend result.
    virtual void UpdateFrontendOutputData(const float time_stamp_s) = 0;

    // Reference for member variables.
    FrontendOptions &options() { return options_; }

    std::unique_ptr<feature_detector::FeaturePointDetector<feature_detector::FastFeature>> &feature_detector() { return feature_detector_; }
    std::unique_ptr<feature_tracker::OpticalFlow> &feature_tracker() { return feature_tracker_; }
    std::vector<std::unique_ptr<sensor_model::CameraBasic>> &camera_models() { return camera_models_; }
    std::unique_ptr<vision_geometry::EpipolarSolver> &epipolar_solver() { return epipolar_solver_; }
    std::unique_ptr<image_processor::ImageProcessor> &image_processor() { return image_processor_; }

    uint8_t *&stored_buff() { return stored_buff_; }
    std::vector<uint8_t> &tracked_status() { return tracked_status_; }

    ImagePyramid *&ref_pyramid_left() { return ref_pyramid_left_; }
    ImagePyramid *&ref_pyramid_right() { return ref_pyramid_right_; }
    std::vector<uint32_t> *&ref_ids() { return ref_ids_; }
    std::vector<Vec2> *&ref_pixel_uv_left() { return ref_pixel_uv_left_; }
    std::vector<Vec2> *&ref_norm_xy_left() { return ref_norm_xy_left_; }
    std::vector<Vec2> *&ref_pixel_uv_right() { return ref_pixel_uv_right_; }
    std::vector<Vec2> *&ref_norm_xy_right() { return ref_norm_xy_right_; }
    std::vector<Vec3> *&ref_points_xyz() { return ref_points_xyz_; }
    std::vector<Vec2> *&ref_vel() { return ref_vel_; }
    std::vector<uint32_t> *&ref_tracked_cnt() { return ref_tracked_cnt_; }

    ImagePyramid *&cur_pyramid_left() { return cur_pyramid_left_; }
    ImagePyramid *&cur_pyramid_right() { return cur_pyramid_right_; }
    std::vector<uint32_t> *&cur_ids() { return cur_ids_; }
    std::vector<Vec2> *&cur_pixel_uv_left() { return cur_pixel_uv_left_; }
    std::vector<Vec2> *&cur_norm_xy_left() { return cur_norm_xy_left_; }
    std::vector<Vec2> *&cur_pixel_uv_right() { return cur_pixel_uv_right_; }
    std::vector<Vec2> *&cur_norm_xy_right() { return cur_norm_xy_right_; }
    std::vector<Vec3> *&cur_points_xyz() { return cur_points_xyz_; }
    std::vector<Vec2> *&cur_vel() { return cur_vel_; }
    std::vector<uint32_t> *&cur_tracked_cnt() { return cur_tracked_cnt_; }

    uint32_t &feature_id_cnt() { return feature_id_cnt_; }
    bool &is_cur_image_keyframe() { return is_cur_image_keyframe_; }
    VisualPointsMeasure &output_data() { return output_data_; }
    slam_data_log::BinaryDataLog &logger() { return logger_; }

    // Const reference for member variables.
    const FrontendOptions &options() const { return options_; }

    const std::unique_ptr<feature_detector::FeaturePointDetector<feature_detector::FastFeature>> &feature_detector() const { return feature_detector_; }
    const std::unique_ptr<feature_tracker::OpticalFlow> &feature_tracker() const { return feature_tracker_; }
    const std::vector<std::unique_ptr<sensor_model::CameraBasic>> &camera_models() const { return camera_models_; }
    const std::unique_ptr<vision_geometry::EpipolarSolver> &epipolar_solver() const { return epipolar_solver_; }
    const std::unique_ptr<image_processor::ImageProcessor> &image_processor() const { return image_processor_; }

    const uint8_t *stored_buff() const { return stored_buff_; }
    const std::vector<uint8_t> &tracked_status() const { return tracked_status_; }

    const ImagePyramid *ref_pyramid_left() const { return ref_pyramid_left_; }
    const ImagePyramid *ref_pyramid_right() const { return ref_pyramid_right_; }
    const std::vector<uint32_t> *ref_ids() const { return ref_ids_; }
    const std::vector<Vec2> *ref_pixel_uv_left() const { return ref_pixel_uv_left_; }
    const std::vector<Vec2> *ref_norm_xy_left() const { return ref_norm_xy_left_; }
    const std::vector<Vec2> *ref_pixel_uv_right() const { return ref_pixel_uv_right_; }
    const std::vector<Vec2> *ref_norm_xy_right() const { return ref_norm_xy_right_; }
    const std::vector<Vec3> *ref_points_xyz() const { return ref_points_xyz_; }
    const std::vector<Vec2> *ref_vel() const { return ref_vel_; }
    const std::vector<uint32_t> *ref_tracked_cnt() const { return ref_tracked_cnt_; }

    const ImagePyramid *cur_pyramid_left() const { return cur_pyramid_left_; }
    const ImagePyramid *cur_pyramid_right() const { return cur_pyramid_right_; }
    const std::vector<uint32_t> *cur_ids() const { return cur_ids_; }
    const std::vector<Vec2> *cur_pixel_uv_left() const { return cur_pixel_uv_left_; }
    const std::vector<Vec2> *cur_norm_xy_left() const { return cur_norm_xy_left_; }
    const std::vector<Vec2> *cur_pixel_uv_right() const { return cur_pixel_uv_right_; }
    const std::vector<Vec2> *cur_norm_xy_right() const { return cur_norm_xy_right_; }
    const std::vector<Vec3> *cur_points_xyz() const { return cur_points_xyz_; }
    const std::vector<Vec2> *cur_vel() const { return cur_vel_; }
    const std::vector<uint32_t> *cur_tracked_cnt() const { return cur_tracked_cnt_; }

    const uint32_t &feature_id_cnt() const { return feature_id_cnt_; }
    const bool &is_cur_image_keyframe() const { return is_cur_image_keyframe_; }
    const VisualPointsMeasure &output_data() const { return output_data_; }
    const slam_data_log::BinaryDataLog &logger() const { return logger_; }

    // Check every components.
    virtual bool CheckAllComponents();

private:
    // Options.
    FrontendOptions options_;

    // Components.
    std::unique_ptr<feature_detector::FeaturePointDetector<feature_detector::FastFeature>> feature_detector_ = nullptr;
    std::unique_ptr<feature_tracker::OpticalFlow> feature_tracker_ = nullptr;
    std::vector<std::unique_ptr<sensor_model::CameraBasic>> camera_models_;
    std::unique_ptr<vision_geometry::EpipolarSolver> epipolar_solver_ = nullptr;
    std::unique_ptr<image_processor::ImageProcessor> image_processor_ = nullptr;

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

    // Outpu data.
    bool is_cur_image_keyframe_ = false;
    VisualPointsMeasure output_data_;

    // Record log.
    slam_data_log::BinaryDataLog logger_;
};

}  // namespace visual_frontend

#endif  // end of _VISUAL_FRONTEND_H_
