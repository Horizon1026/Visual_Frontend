#include "frontend_stereo.h"
#include "slam_operations.h"
#include "log_report.h"
#include "tick_tock.h"
#include "visualizor.h"

namespace VISUAL_FRONTEND {

constexpr uint32_t kFrontendStereoCurvesLogIndex = 1;
constexpr uint32_t kFrontendStereoTrackingResultIndex = 2;
constexpr int32_t kMaxAllowedNonEpipolarDirectionPixelResidual = 20;

bool FrontendStereo::ProcessSourceImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    RETURN_FALSE_IF(cur_image_left.data() == nullptr || cur_image_right.data() == nullptr);

    if (image_processor() == nullptr) {
        std::copy_n(cur_image_left.data(), cur_image_left.rows() * cur_image_left.cols(), cur_pyramid_left()->GetImage(0).data());
        std::copy_n(cur_image_right.data(), cur_image_right.rows() * cur_image_right.cols(), cur_pyramid_right()->GetImage(0).data());
    } else {
        image_processor()->Process(cur_image_left, cur_pyramid_left()->GetImage(0));
        image_processor()->Process(cur_image_right, cur_pyramid_right()->GetImage(0));
    }

    cur_pyramid_left()->CreateImagePyramid(4);
    cur_pyramid_right()->CreateImagePyramid(4);

    return true;
}

bool FrontendStereo::PredictPixelLocationInCurrentLeftImage() {
    *cur_pixel_uv_left() = *ref_pixel_uv_left();
    for (uint32_t i = 0; i < ref_vel()->size(); ++i) {
        (*cur_pixel_uv_left())[i] += (*ref_vel())[i];
    }
    return true;
}

bool FrontendStereo::TrackFeaturesFromRefernceLeftToCurrentLeftImage() {
    *cur_ids() = *ref_ids();
    tracked_status().clear();
    if (!feature_tracker()->TrackFeatures(*ref_pyramid_left(), *cur_pyramid_left(), *ref_pixel_uv_left(), *cur_pixel_uv_left(), tracked_status())) {
        ReportError("feature_tracker()->TrackFeatures track from ref_left to cur_left error.");
        return false;
    }

    // Record log data.
    log_package_data_.num_of_old_features_in_only_left = ref_pixel_uv_left()->size();
    log_package_data_.num_of_tracked_features_in_only_left = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

    return true;
}

bool FrontendStereo::RejectOutliersBetweenRefernceLeftAndCurrentLeftImage() {
    Mat3 essential;
    cur_norm_xy_left()->resize(cur_pixel_uv_left()->size());
    for (uint32_t i = 0; i < cur_pixel_uv_left()->size(); ++i) {
        camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left())[i], (*cur_norm_xy_left())[i]);
    }

    if (epipolar_solver() == nullptr) {
        // If epipolar solver is not given, do outliers rejection by klt tracking back.
        ref_pixel_uv_left_tracked_back_ = *cur_pixel_uv_left();
        for (uint32_t i = 0; i < ref_vel()->size(); ++i) {
            ref_pixel_uv_left_tracked_back_[i] -= (*ref_vel())[i];
        }

        if (!feature_tracker()->TrackFeatures(*cur_pyramid_left(), *ref_pyramid_left(), *cur_pixel_uv_left(), ref_pixel_uv_left_tracked_back_, tracked_status())) {
            ReportError("feature_tracker()->TrackFeatures error.");
            return false;
        }

        for (uint32_t i = 0; i < tracked_status().size(); ++i) {
            if (tracked_status()[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                if (((*ref_pixel_uv_left())[i] - ref_pixel_uv_left_tracked_back_[i]).squaredNorm() > options().kMaxValidTrackBackPixelResidual) {
                    tracked_status()[i] = static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kLargeResidual);
                }
            }
        }
    } else {
        // If epipolar solver is given, do outliers rejection by it.
        if (!epipolar_solver()->EstimateEssential(*ref_norm_xy_left(), *cur_norm_xy_left(), essential, tracked_status())) {
            ReportError("epipolar_solver()->EstimateEssential error");
            return false;
        }
    }

    // Record log data.
    log_package_data_.num_of_inliers_in_only_left = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendStereo::ComputeOpticalFlowVelocity() {
    cur_vel()->resize(ref_pixel_uv_left()->size());
    for (uint32_t i = 0; i < ref_pixel_uv_left()->size(); ++i) {
        if (tracked_status()[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            (*cur_vel())[i] = (*cur_pixel_uv_left())[i] - (*ref_pixel_uv_left())[i];
        } else {
            (*cur_vel())[i].setZero();
        }
    }
    *ref_vel() = *cur_vel();
    return true;
}

bool FrontendStereo::SparsifyTrackedFeaturesInLeft() {
    feature_detector()->SparsifyFeatures(*cur_pixel_uv_left(),
                                            cur_pyramid_left()->GetImage(0).rows(),
                                            cur_pyramid_left()->GetImage(0).cols(),
                                            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
                                            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kNotTracked),
                                            tracked_status());
    const int32_t num_of_inliers_after_filter_in_only_left = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

    // Record log data.
    log_package_data_.num_of_inliers_after_filter_in_only_left = num_of_inliers_after_filter_in_only_left;
    return true;
}

bool FrontendStereo::TrackFeaturesFromCurrentLeftToCurrentRightImage(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    *cur_stereo_tracked_status_ = tracked_status();
    *cur_pixel_uv_right() = *cur_pixel_uv_left();

    const int32_t stored_half_row_size = feature_tracker()->options().kPatchRowHalfSize;
    const int32_t stored_half_col_size = feature_tracker()->options().kPatchColHalfSize;
    feature_tracker()->options().kPatchRowHalfSize = half_patch_size_for_stereo_tracking_.x();
    feature_tracker()->options().kPatchColHalfSize = half_patch_size_for_stereo_tracking_.y();
    if (!feature_tracker()->TrackFeatures(*cur_pyramid_left(), *cur_pyramid_right(), *cur_pixel_uv_left(), *cur_pixel_uv_right(), *cur_stereo_tracked_status_)) {
        ReportError("feature_tracker()->TrackFeatures track from cur_left to cur_right error.");
        return false;
    }
    feature_tracker()->options().kPatchRowHalfSize = stored_half_row_size;
    feature_tracker()->options().kPatchColHalfSize = stored_half_col_size;

    // Record log data.
    log_package_data_.num_of_tracked_feature_from_left_to_right = SlamOperation::StatisItemInVector(*cur_stereo_tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendStereo::RejectOutliersBetweenCurrentLeftToCurrentRightImage() {
    cur_norm_xy_right()->resize(cur_pixel_uv_right()->size());
    for (uint32_t i = 0; i < cur_pixel_uv_right()->size(); ++i) {
        camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_right())[i], (*cur_norm_xy_right())[i]);
    }

    if (epipolar_solver() == nullptr) {
        // If epipolar solver is not given, do outliers rejection by base line.
        RETURN_FALSE_IF(cur_norm_xy_left()->size() != cur_norm_xy_right()->size());
        RETURN_FALSE_IF(cur_stereo_tracked_status_->size() != cur_norm_xy_right()->size());
        for (uint32_t i = 0; i < cur_stereo_tracked_status_->size(); ++i) {
            if (std::fabs((*cur_norm_xy_left())[i].y() - (*cur_norm_xy_right())[i].y()) > kMaxAllowedNonEpipolarDirectionPixelResidual / camera_model()->fy()) {
                (*cur_stereo_tracked_status_)[i] = static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kLargeResidual);
            }
        }
    } else {
        // If epipolar solver is given, do outliers rejection by it.
        Mat3 essential;
        if (!epipolar_solver()->EstimateEssential(*cur_norm_xy_left(), *cur_norm_xy_right(), essential, *cur_stereo_tracked_status_)) {
            ReportError("epipolar_solver()->EstimateEssential error");
            return false;
        }
    }

    // Record log data.
    log_package_data_.num_of_inliers_from_left_to_right = SlamOperation::StatisItemInVector(*cur_stereo_tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendStereo::SelectKeyframe() {
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    is_cur_image_keyframe() = tracked_num < options().kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options().kSelfSelectKeyframe;
    // Record log data.
    log_package_data_.is_keyframe = is_cur_image_keyframe();
    return true;
}

bool FrontendStereo::AdjustTrackingResultByStatus() {
    // Update tracked statis result.
    for (uint32_t i = 0; i < tracked_status().size(); ++i) {
        if (tracked_status()[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            ++(*ref_tracked_cnt())[i];
        }
    }

    // Adjust result.
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_pixel_uv_left());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_pixel_uv_right());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_ids());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_norm_xy_left());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_norm_xy_right());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_vel());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *ref_tracked_cnt());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_stereo_tracked_status_);
    tracked_status().resize(cur_pixel_uv_left()->size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

    return true;
}

bool FrontendStereo::SupplememtNewFeatures(const GrayImage &cur_image_left) {
    feature_detector()->DetectGoodFeatures(cur_image_left,
                                           options().kMaxStoredFeaturePointsNumber,
                                           *cur_pixel_uv_left());
    const uint32_t old_features_num = cur_ids()->size();
    const uint32_t new_features_num = cur_pixel_uv_left()->size() - old_features_num;
    Vec2 temp_cur_norm_xy_left = Vec2::Zero();

    for (uint32_t i = 0; i < new_features_num; ++i) {
        cur_ids()->emplace_back(feature_id_cnt());

        camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left())[i + old_features_num], temp_cur_norm_xy_left);
        cur_norm_xy_left()->emplace_back(temp_cur_norm_xy_left);

        ref_tracked_cnt()->emplace_back(1);
        ++feature_id_cnt();
    }
    // Record log data.
    log_package_data_.num_of_new_features_in_only_left = new_features_num;
    return true;
}

bool FrontendStereo::MakeCurrentFrameKeyframe() {
    // When current frame becomes keyframe, prediction will not work in next tracking.
    cur_vel()->clear();

    // Replace ref with cur.
    SlamOperation::ExchangePointer(&ref_pyramid_left(), &cur_pyramid_left());
    SlamOperation::ExchangePointer(&ref_pyramid_right(), &cur_pyramid_right());
    SlamOperation::ExchangePointer(&ref_pixel_uv_left(), &cur_pixel_uv_left());
    SlamOperation::ExchangePointer(&ref_pixel_uv_right(), &cur_pixel_uv_right());
    SlamOperation::ExchangePointer(&ref_ids(), &cur_ids());
    SlamOperation::ExchangePointer(&ref_norm_xy_left(), &cur_norm_xy_left());
    SlamOperation::ExchangePointer(&ref_norm_xy_right(), &cur_norm_xy_right());
    SlamOperation::ExchangePointer(&ref_vel(), &cur_vel());
    SlamOperation::ExchangePointer(&ref_stereo_tracked_status_, &cur_stereo_tracked_status_);

    return true;
}

bool FrontendStereo::RunOnce(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());
    // GrayImage process.
    RETURN_FALSE_IF_FALSE(ProcessSourceImage(cur_image_left, cur_image_right));

    // Track features if ref frame is ok.
    if (!ref_pixel_uv_left()->empty()) {
        // Predict pixel location on current image by optical flow velocity.
        RETURN_FALSE_IF_FALSE(PredictPixelLocationInCurrentLeftImage());
        // Track features from ref pyramid to cur pyramid.
        RETURN_FALSE_IF_FALSE(TrackFeaturesFromRefernceLeftToCurrentLeftImage());
        // Reject outliers by essential/fundemantal matrix.
        RETURN_FALSE_IF_FALSE(RejectOutliersBetweenRefernceLeftAndCurrentLeftImage());
        // Compute optical flow velocity. It is useful for feature prediction.
        RETURN_FALSE_IF_FALSE(ComputeOpticalFlowVelocity());
        // Grid filter to make points sparsely.
        RETURN_FALSE_IF_FALSE(SparsifyTrackedFeaturesInLeft());
        // Track features from cur pyramid left to cur pyramid right.
        RETURN_FALSE_IF_FALSE(TrackFeaturesFromCurrentLeftToCurrentRightImage(cur_image_left, cur_image_right));
        // Reject outliers by essential/fundemantal matrix.
        RETURN_FALSE_IF_FALSE(RejectOutliersBetweenCurrentLeftToCurrentRightImage());
    }

    // Check if cur_pyarmid should be keyframe.
    RETURN_FALSE_IF_FALSE(SelectKeyframe());

    // Visualize result when this API is defined.
    DrawTrackingResults("Frontend Stereo Tracking Result");

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe()) {
        // Prepare to make current frame to be reference frame (keyframe).
        RETURN_FALSE_IF_FALSE(AdjustTrackingResultByStatus());
        // Detect new features in cur.
        RETURN_FALSE_IF_FALSE(SupplememtNewFeatures(cur_image_left));
        // Current frame becomes keyframe.
        RETURN_FALSE_IF_FALSE(MakeCurrentFrameKeyframe());

    } else {
        // Record log data.
        log_package_data_.num_of_new_features_in_only_left = 0;
    }

    // Update frontend output data.
    UpdateFrontendOutputData();

    // Record package data.
    if (options().kEnableRecordBinaryLog) {
        logger().RecordPackage(kFrontendStereoCurvesLogIndex, reinterpret_cast<const char *>(&log_package_data_));
    }

    return true;
}

// Draw tracking results.
void FrontendStereo::DrawTrackingResults(const std::string &title) {
    if (!options().kEnableVisualizeResult) {
        return;
    }

    RgbImage show_image;
    Visualizor::DrawImageWithTrackedFeaturesWithId(
        ref_pyramid_left()->GetImage(0),
        ref_pyramid_right()->GetImage(0),
        cur_pyramid_left()->GetImage(0),
        cur_pyramid_right()->GetImage(0),
        *ref_pixel_uv_left(),
        *ref_pixel_uv_right(),
        *cur_pixel_uv_left(),
        *cur_pixel_uv_right(),
        *ref_ids(),
        *ref_ids(),
        *cur_ids(),
        *cur_ids(),
        tracked_status(),
        *ref_stereo_tracked_status_,
        *cur_stereo_tracked_status_,
        static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
        *ref_tracked_cnt(),
        *cur_vel(),
        show_image
    );
    Visualizor::ShowImage(title, show_image);
    Visualizor::WaitKey(1);

    if (options().kEnableRecordBinaryLog) {
        logger().RecordPackage(kFrontendStereoTrackingResultIndex, show_image);
    }
}

// Support for log recording.
void FrontendStereo::RegisterLogPackages() {
    using namespace SLAM_DATA_LOG;

    std::unique_ptr<PackageInfo> package_ptr = std::make_unique<PackageInfo>();
    package_ptr->id = kFrontendStereoCurvesLogIndex;
    package_ptr->name = "frontend_stereo";
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_keyframe"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_old_features_in_only_left"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_tracked_features_in_only_left"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_inliers_in_only_left"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_inliers_after_filter_in_only_left"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_tracked_feature_from_left_to_right"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_inliers_from_left_to_right"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_new_features_in_only_left"});
    if (!logger().RegisterPackage(package_ptr)) {
        ReportError("Failed to register package.");
    }

    package_ptr = std::make_unique<PackageInfo>();
    package_ptr->id = kFrontendStereoTrackingResultIndex;
    package_ptr->name = "frontend_stereo_result";
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kImage, .name = "track_result"});
    if (!logger().RegisterPackage(package_ptr)) {
        ReportError("Failed to register package.");
    }
}

// Update frontend result.
void FrontendStereo::UpdateFrontendOutputData() {
    output_data().features_id.clear();
    output_data().observes_per_frame.clear();

    output_data().is_current_keyframe = is_cur_image_keyframe();
    if (output_data().is_current_keyframe) {
        // If current frame is keyframe, tracking result will be stored in ref_info.
        output_data().features_id = *ref_ids();
        output_data().tracked_cnt = *ref_tracked_cnt();
        for (uint32_t i = 0; i < ref_ids()->size(); ++i) {
            output_data().observes_per_frame.emplace_back(ObservePerFrame { ObservePerView {
                .id = 0,
                .raw_pixel_uv = (*ref_pixel_uv_left())[i],
                .rectified_norm_xy = (*ref_norm_xy_left())[i],
            }});
        }
        for (uint32_t i = 0; i < ref_stereo_tracked_status_->size(); ++i) {
            if ((*ref_stereo_tracked_status_)[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                output_data().observes_per_frame[i].emplace_back(ObservePerView {
                    .id = 1,
                    .raw_pixel_uv = (*ref_pixel_uv_right())[i],
                    .rectified_norm_xy = (*ref_norm_xy_right())[i],
                });
            }
        }

    } else {
        // If current frame is not keyframe, tracking result will be stored in cur_info.
        output_data().features_id = *cur_ids();
        output_data().tracked_cnt = *cur_tracked_cnt();
        for (uint32_t i = 0; i < cur_ids()->size(); ++i) {
            output_data().observes_per_frame.emplace_back(ObservePerFrame { ObservePerView {
                .id = 0,
                .raw_pixel_uv = (*cur_pixel_uv_left())[i],
                .rectified_norm_xy = (*cur_norm_xy_left())[i],
            }});
        }
        for (uint32_t i = 0; i < cur_stereo_tracked_status_->size(); ++i) {
            if ((*cur_stereo_tracked_status_)[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                output_data().observes_per_frame[i].emplace_back(ObservePerView {
                    .id = 1,
                    .raw_pixel_uv = (*cur_pixel_uv_right())[i],
                    .rectified_norm_xy = (*cur_norm_xy_right())[i],
                });
            }
        }
    }
}

}
