#include "frontend_stereo.h"
#include "slam_operations.h"
#include "log_report.h"
#include "tick_tock.h"
#include "visualizor.h"

namespace VISUAL_FRONTEND {

bool FrontendStereo::ProcessSourceImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    RETURN_FALSE_IF(cur_image_left.data() == nullptr || cur_image_right.data() == nullptr);

    if (image_processor_ == nullptr) {
        std::copy_n(cur_image_left.data(), cur_image_left.rows() * cur_image_left.cols(), cur_pyramid_left_->GetImage(0).data());
        std::copy_n(cur_image_right.data(), cur_image_right.rows() * cur_image_right.cols(), cur_pyramid_right_->GetImage(0).data());
    } else {
        image_processor_->Process(cur_image_left, cur_pyramid_left_->GetImage(0));
        image_processor_->Process(cur_image_right, cur_pyramid_right_->GetImage(0));
    }

    cur_pyramid_left_->CreateImagePyramid(4);
    cur_pyramid_right_->CreateImagePyramid(4);

    return true;
}

bool FrontendStereo::RunOnce(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    ReportInfo("---------------------------------------------------------");

    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());

    // GrayImage process.
    RETURN_FALSE_IF_FALSE(ProcessSourceImage(cur_image_left, cur_image_right));

    // Track features if ref frame is ok.
    if (ref_pixel_uv_left_->size() != 0) {
        // Predict pixel location on current image by optical flow velocity.
        *cur_pixel_uv_left_ = *ref_pixel_uv_left_;    // Deep copy.
        for (uint32_t i = 0; i < ref_vel_->size(); ++i) {
            (*cur_pixel_uv_left_)[i] += (*ref_vel_)[i];
        }

        // Track features from ref pyramid to cur pyramid.
        *cur_ids_ = *ref_ids_;
        tracked_status_.clear();
        if (!feature_tracker_->TrackFeatures(*ref_pyramid_left_, *cur_pyramid_left_, *ref_pixel_uv_left_, *cur_pixel_uv_left_, tracked_status_)) {
            ReportError("feature_tracker_->TrackFeatures track from ref_left to cur_left error.");
            return false;
        }
        ReportInfo("After optical flow tracking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status_.size());

        // Reject outliers by essential/fundemantal matrix.
        Mat3 essential;
        cur_norm_xy_left_->resize(cur_pixel_uv_left_->size());
        for (uint32_t i = 0; i < cur_pixel_uv_left_->size(); ++i) {
            camera_model_->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left_)[i], (*cur_norm_xy_left_)[i]);
        }
        if (!epipolar_solver_->EstimateEssential(*ref_norm_xy_left_, *cur_norm_xy_left_, essential, tracked_status_)) {
            ReportError("epipolar_solver_->EstimateEssential error");
            return false;
        }
        ReportInfo("After essential checking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status_.size());

        // Compute optical flow velocity. It is useful for feature prediction.
        cur_vel_->resize(ref_pixel_uv_left_->size());
        for (uint32_t i = 0; i < ref_pixel_uv_left_->size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                (*cur_vel_)[i] = (*cur_pixel_uv_left_)[i] - (*ref_pixel_uv_left_)[i];
            } else {
                (*cur_vel_)[i].setZero();
            }
        }
        *ref_vel_ = *cur_vel_;

        // Track features from cur pyramid left to cur pyramid right.
        *cur_pixel_uv_right_ = *cur_pixel_uv_left_;
        // Adjust patch size for stereo tracking.
        const int32_t stored_half_row_size = feature_tracker_->options().kPatchRowHalfSize;
        const int32_t stored_half_col_size = feature_tracker_->options().kPatchColHalfSize;
        feature_tracker_->options().kPatchRowHalfSize = 1;
        feature_tracker_->options().kPatchColHalfSize = 25;
        if (!feature_tracker_->TrackFeatures(*cur_pyramid_left_, *cur_pyramid_right_, *cur_pixel_uv_left_, *cur_pixel_uv_right_, tracked_status_)) {
            ReportError("feature_tracker_->TrackFeatures track from cur_left to cur_right error.");
            return false;
        }
        feature_tracker_->options().kPatchRowHalfSize = stored_half_row_size;
        feature_tracker_->options().kPatchColHalfSize = stored_half_col_size;

        // Reject outliers by essential/fundemantal matrix.
        cur_norm_xy_right_->resize(cur_pixel_uv_right_->size());
        for (uint32_t i = 0; i < cur_pixel_uv_right_->size(); ++i) {
            camera_model_->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_right_)[i], (*cur_norm_xy_right_)[i]);
        }
        if (!epipolar_solver_->EstimateEssential(*cur_norm_xy_left_, *cur_norm_xy_right_, essential, tracked_status_)) {
            ReportError("epipolar_solver_->EstimateEssential error");
            return false;
        }

        // Grid filter to make points sparsely.
        feature_detector_->SparsifyFeatures(*cur_pixel_uv_left_,
                                            cur_pyramid_left_->GetImage(0).rows(),
                                            cur_pyramid_left_->GetImage(0).cols(),
                                            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
                                            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kNotTracked),
                                            tracked_status_);
        ReportInfo("After grid filtering, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status_.size());
    }

    // Check if cur_pyarmid should be keyframe.
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    is_cur_image_keyframe_ = tracked_num < options_.kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options_.kSelfSelectKeyframe;
    if (is_cur_image_keyframe_) {
        ReportInfo("Current frame is keyframe.");
    }

    // Visualize result when this API is defined.
    Visualizor::ShowImageWithTrackedFeaturesWithId(
        "Frontend Mono Tracking Result",
        ref_pyramid_left_->GetImage(0),
        ref_pyramid_right_->GetImage(0),
        cur_pyramid_left_->GetImage(0),
        cur_pyramid_right_->GetImage(0),
        *ref_pixel_uv_left_,
        *ref_pixel_uv_right_,
        *cur_pixel_uv_left_,
        *cur_pixel_uv_right_,
        *ref_ids_,
        *ref_ids_,
        *cur_ids_,
        *cur_ids_,
        *ref_tracked_cnt_,
        *cur_vel_
    );
    Visualizor::WaitKey(1);

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe_) {
        // Prepare to make current frame to be reference frame (keyframe).
        // Update tracked statis result.
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                ++(*ref_tracked_cnt_)[i];
            }
        }

        // Adjust result.
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_pixel_uv_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_pixel_uv_right_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_ids_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_norm_xy_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_norm_xy_right_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_vel_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *ref_tracked_cnt_);
        tracked_status_.resize(cur_pixel_uv_left_->size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

        // Detect new features in cur.
        feature_detector_->DetectGoodFeatures(cur_image_left,
                                              options_.kMaxStoredFeaturePointsNumber,
                                              *cur_pixel_uv_left_);
        const uint32_t new_features_num = cur_pixel_uv_left_->size() - cur_ids_->size();
        for (uint32_t i = 0; i < new_features_num; ++i) {
            cur_ids_->emplace_back(feature_id_cnt_);
            cur_norm_xy_left_->emplace_back(Vec2::Zero());
            cur_norm_xy_right_->emplace_back(Vec2::Zero());
            ref_tracked_cnt_->emplace_back(1);
            ++feature_id_cnt_;
        }

        // Current frame becomes keyframe.
        // When current frame becomes keyframe, prediction will not work in next tracking.
        cur_vel_->clear();

        // Replace ref with cur.
        SlamOperation::ExchangePointer(&ref_pyramid_left_, &cur_pyramid_left_);
        SlamOperation::ExchangePointer(&ref_pyramid_right_, &cur_pyramid_right_);
        SlamOperation::ExchangePointer(&ref_pixel_uv_left_, &cur_pixel_uv_left_);
        SlamOperation::ExchangePointer(&ref_pixel_uv_right_, &cur_pixel_uv_right_);
        SlamOperation::ExchangePointer(&ref_ids_, &cur_ids_);
        SlamOperation::ExchangePointer(&ref_norm_xy_left_, &cur_norm_xy_left_);
        SlamOperation::ExchangePointer(&ref_norm_xy_right_, &cur_norm_xy_right_);
        SlamOperation::ExchangePointer(&ref_vel_, &cur_vel_);
    }

    return true;
}

}
