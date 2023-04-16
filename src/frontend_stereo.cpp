#include "frontend_stereo.h"
#include "slam_operations.h"
#include "log_api.h"

namespace VISUAL_FRONTEND {

bool FrontendStereo::ProcessSourceImage(const Image &cur_image_left, const Image &cur_image_right) {
    RETURN_FALSE_IF(cur_image_left.data() == nullptr || cur_image_right.data() == nullptr);

    std::copy_n(cur_image_left.data(), cur_image_left.rows() * cur_image_left.cols(), cur_pyramid_left_->GetImage(0).data());
    cur_pyramid_left_->CreateImagePyramid(4);
    std::copy_n(cur_image_right.data(), cur_image_right.rows() * cur_image_right.cols(), cur_pyramid_right_->GetImage(0).data());
    cur_pyramid_right_->CreateImagePyramid(4);

    return true;
}

bool FrontendStereo::RunOnce(const Image &cur_image_left, const Image &cur_image_right) {
    LogInfo("---------------------------------------------------------");

    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());

    // Image process.
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
        if (!feature_tracker_->TrackMultipleLevel(*ref_pyramid_left_, *cur_pyramid_left_, *ref_pixel_uv_left_, *cur_pixel_uv_left_, tracked_status_)) {
            LogError("feature_tracker_->TrackMultipleLevel error.");
            return false;
        }

        // Compute optical flow velocity. It is useful for feature prediction.
        cur_vel_->resize(ref_pixel_uv_left_->size());
        for (uint32_t i = 0; i < ref_pixel_uv_left_->size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                (*cur_vel_)[i] = (*cur_pixel_uv_left_)[i] - (*ref_pixel_uv_left_)[i];
            } else {
                (*cur_vel_)[i].setZero();
            }
        }

        LogInfo("After optical flow tracking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED))
            << " / " << tracked_status_.size());

        // Reject outliers by essential/fundemantal matrix.
        cur_norm_xy_left_->resize(cur_pixel_uv_left_->size());
        for (uint32_t i = 0; i < cur_pixel_uv_left_->size(); ++i) {
            camera_model_->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left_)[i], (*cur_norm_xy_left_)[i]);
        }

        Mat3 essential;
        if (!epipolar_solver_->EstimateEssential(*ref_norm_xy_left_, *cur_norm_xy_left_, essential, tracked_status_)) {
            LogError("epipolar_solver_->EstimateEssential error");
            return false;
        }

        // Reject outliers' optical flow velocity. It means do not predict them at next tracking.
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            if (tracked_status_[i] != static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                (*cur_vel_)[i].setZero();
            }
        }

        *ref_vel_ = *cur_vel_;
        LogInfo("After essential checking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED))
            << " / " << tracked_status_.size());

        // Track features from cur pyramid left to cur pyramid right.
        // TODO:

        // Grid filter to make points sparsely.
        feature_detector_->SparsifyFeatures(*cur_pixel_uv_left_,
                                            cur_pyramid_left_->GetImage(0).rows(),
                                            cur_pyramid_left_->GetImage(0).cols(),
                                            static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED),
                                            static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::NOT_TRACKED),
                                            tracked_status_);
        LogInfo("After grid filtering, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED))
            << " / " << tracked_status_.size());
    }

    // Check if cur_pyarmid should be keyframe.
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED));
    is_cur_image_keyframe_ = tracked_num < options_.kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options_.kSelfSelectKeyframe;
    if (is_cur_image_keyframe_) {
        LogInfo("Current frame is keyframe.");
    }

    // Visualize result when this API is defined.
    if (VisualizeResult != nullptr) {
        VisualizeResult();
    }

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe_) {
        // Prepare to make current frame to be reference frame (keyframe).
        // Update tracked statis result.
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                ++(*ref_tracked_cnt_)[i];
            }
        }

        // Adjust result.
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_pixel_uv_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_ids_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_norm_xy_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_vel_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *ref_tracked_cnt_);
        tracked_status_.resize(cur_pixel_uv_left_->size(), static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED));

        // Detect new features in cur.
        feature_detector_->DetectGoodFeatures(cur_pyramid_left_->GetImage(0),
                                              options_.kMaxStoredFeaturePointsNumber,
                                              *cur_pixel_uv_left_);
        const uint32_t new_features_num = cur_pixel_uv_left_->size() - cur_ids_->size();
        for (uint32_t i = 0; i < new_features_num; ++i) {
            cur_ids_->emplace_back(feature_id_cnt_);
            cur_norm_xy_left_->emplace_back(Vec2::Zero());
            ref_tracked_cnt_->emplace_back(1);
            ++feature_id_cnt_;
        }

        // Current frame becomes keyframe.
        // When current frame becomes keyframe, prediction will not work in next tracking.
        cur_vel_->clear();

        // Replace ref with cur.
        SlamOperation::ExchangePointer(&ref_pyramid_left_, &cur_pyramid_left_);
        SlamOperation::ExchangePointer(&ref_pixel_uv_left_, &cur_pixel_uv_left_);
        SlamOperation::ExchangePointer(&ref_ids_, &cur_ids_);
        SlamOperation::ExchangePointer(&ref_norm_xy_left_, &cur_norm_xy_left_);
        SlamOperation::ExchangePointer(&ref_vel_, &cur_vel_);
    }

    return true;
}

}
