#include "frontend_mono.h"
#include "slam_operations.h"
#include "log_report.h"
#include "tick_tock.h"
#include "visualizor.h"

namespace VISUAL_FRONTEND {

constexpr uint32_t kFrontendMonoLogIndex = 1;

bool FrontendMono::ProcessSourceImage(const GrayImage &cur_image) {
    RETURN_FALSE_IF(cur_image.data() == nullptr);

    if (image_processor_ == nullptr) {
        std::copy_n(cur_image.data(), cur_image.rows() * cur_image.cols(), cur_pyramid_left_->GetImage(0).data());
    } else {
        image_processor_->Process(cur_image, cur_pyramid_left_->GetImage(0));
    }

    cur_pyramid_left_->CreateImagePyramid(4);
    return true;
}

bool FrontendMono::PredictPixelLocation() {
    *cur_pixel_uv_left_ = *ref_pixel_uv_left_;    // Deep copy.
    for (uint32_t i = 0; i < ref_vel_->size(); ++i) {
        (*cur_pixel_uv_left_)[i] += (*ref_vel_)[i];
    }
    return true;
}

bool FrontendMono::TrackFeatures() {
    *cur_ids_ = *ref_ids_;
    tracked_status_.clear();
    if (!feature_tracker_->TrackFeatures(*ref_pyramid_left_, *cur_pyramid_left_, *ref_pixel_uv_left_, *cur_pixel_uv_left_, tracked_status_)) {
        ReportError("feature_tracker_->TrackFeatures error.");
        return false;
    }

    // Record log data.
    log_package_data_.num_of_old_features = tracked_status_.size();
    log_package_data_.num_of_tracked_features = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem() {
    cur_norm_xy_left_->resize(cur_pixel_uv_left_->size());
    for (uint32_t i = 0; i < cur_pixel_uv_left_->size(); ++i) {
        camera_model_->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left_)[i], (*cur_norm_xy_left_)[i]);
    }

    return true;
}

bool FrontendMono::RejectOutliersByEpipolarConstrain() {
    Mat3 essential;
    if (!epipolar_solver_->EstimateEssential(*ref_norm_xy_left_, *cur_norm_xy_left_, essential, tracked_status_)) {
        ReportError("epipolar_solver_->EstimateEssential error");
        return false;
    }

    // Record log data.
    log_package_data_.num_of_inliers = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::RejectOutliersByTrackingBack() {
    ref_pixel_xy_left_tracked_back_ = *cur_pixel_uv_left_;

    for (uint32_t i = 0; i < ref_vel_->size(); ++i) {
        ref_pixel_xy_left_tracked_back_[i] -= (*ref_vel_)[i];
    }

    if (!feature_tracker_->TrackFeatures(*cur_pyramid_left_, *ref_pyramid_left_, *cur_pixel_uv_left_, ref_pixel_xy_left_tracked_back_, tracked_status_)) {
        ReportError("feature_tracker_->TrackFeatures error.");
        return false;
    }

    for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
        if (tracked_status_[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            if (((*ref_pixel_uv_left_)[i] - ref_pixel_xy_left_tracked_back_[i]).squaredNorm() > options_.kMaxValidTrackBackPixelResidual) {
                tracked_status_[i] = static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kLargeResidual);
            }
        }
    }

    // Record log data.
    log_package_data_.num_of_inliers = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::ComputeOpticalFlowVelocity() {
    cur_vel_->resize(ref_pixel_uv_left_->size());
    for (uint32_t i = 0; i < ref_pixel_uv_left_->size(); ++i) {
        if (tracked_status_[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            (*cur_vel_)[i] = (*cur_pixel_uv_left_)[i] - (*ref_pixel_uv_left_)[i];
        } else {
            (*cur_vel_)[i].setZero();
        }
    }

    *ref_vel_ = *cur_vel_;

    return true;
}

bool FrontendMono::SparsifyTrackedFeatures() {
    feature_detector_->SparsifyFeatures(*cur_pixel_uv_left_,
                                        cur_pyramid_left_->GetImage(0).rows(),
                                        cur_pyramid_left_->GetImage(0).cols(),
                                        static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
                                        static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kNotTracked),
                                        tracked_status_);

    // Record log data.
    log_package_data_.num_of_inliers_after_filter = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::SelectKeyframe() {
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status_, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    is_cur_image_keyframe_ = tracked_num < options_.kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options_.kSelfSelectKeyframe;

    // Record log data.
    log_package_data_.is_keyframe = static_cast<uint8_t>(is_cur_image_keyframe_);
    return true;
}

bool FrontendMono::AdjustTrackingResultByStatus() {
    // Update tracked statis result.
    for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
        if (tracked_status_[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            ++(*ref_tracked_cnt_)[i];
        }
    }

    // Adjust result.
    SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_pixel_uv_left_);
    SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_ids_);
    SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_norm_xy_left_);
    SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_vel_);
    SlamOperation::ReduceVectorByStatus(tracked_status_, *ref_tracked_cnt_);
    tracked_status_.resize(cur_pixel_uv_left_->size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

    return true;
}

bool FrontendMono::SupplementNewFeatures(const GrayImage &cur_image_left) {
    feature_detector_->DetectGoodFeatures(cur_image_left,
                                          options_.kMaxStoredFeaturePointsNumber,
                                          *cur_pixel_uv_left_);

    const uint32_t new_features_num = cur_pixel_uv_left_->size() - cur_ids_->size();

    for (uint32_t i = 0; i < new_features_num; ++i) {
        cur_ids_->emplace_back(feature_id_cnt_);
        cur_norm_xy_left_->emplace_back(Vec2::Zero());
        ref_tracked_cnt_->emplace_back(1);
        ++feature_id_cnt_;
    }

    // Record log data.
    log_package_data_.num_of_new_features = new_features_num;
    return true;
}

bool FrontendMono::MakeCurrentFrameKeyframe() {
    // When current frame becomes keyframe, prediction will not work in next tracking.
    cur_vel_->clear();

    // Replace ref with cur.
    SlamOperation::ExchangePointer(&ref_pyramid_left_, &cur_pyramid_left_);
    SlamOperation::ExchangePointer(&ref_pixel_uv_left_, &cur_pixel_uv_left_);
    SlamOperation::ExchangePointer(&ref_ids_, &cur_ids_);
    SlamOperation::ExchangePointer(&ref_norm_xy_left_, &cur_norm_xy_left_);
    SlamOperation::ExchangePointer(&ref_vel_, &cur_vel_);

    return true;
}

bool FrontendMono::RunOnce(const GrayImage &cur_image) {
    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());
    // GrayImage process.
    RETURN_FALSE_IF_FALSE(ProcessSourceImage(cur_image));

    // Track features if ref frame is ok.
    if (!ref_pixel_uv_left_->empty()) {
        // Predict pixel location on current image by optical flow velocity.
        RETURN_FALSE_IF_FALSE(PredictPixelLocation());
        // Track features from ref pyramid to cur pyramid.
        RETURN_FALSE_IF_FALSE(TrackFeatures());
        // Lift and do undistortion.
        RETURN_FALSE_IF_FALSE(LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem());

        if (epipolar_solver_ == nullptr) {
            // Reject outliers by essential/fundemantal matrix.
            RETURN_FALSE_IF_FALSE(RejectOutliersByTrackingBack());
        } else {
            // Reject outliers by tracking back.
            RETURN_FALSE_IF_FALSE(RejectOutliersByEpipolarConstrain());
        }

        // Compute optical flow velocity. It is useful for feature prediction.
        RETURN_FALSE_IF_FALSE(ComputeOpticalFlowVelocity());
        // Grid filter to make points sparsely.
        RETURN_FALSE_IF_FALSE(SparsifyTrackedFeatures());
    }

    // Check if cur_pyarmid should be keyframe.
    RETURN_FALSE_IF_FALSE(SelectKeyframe());

    // Visualize result when this API is defined.
    DrawTrackingResults("Frontend Mono Tracking Result");

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe_) {
        // Prepare to make current frame to be reference frame (keyframe).
        RETURN_FALSE_IF_FALSE(AdjustTrackingResultByStatus());
        // Detect new features in cur.
        RETURN_FALSE_IF_FALSE(SupplementNewFeatures(cur_image));
        // Current frame becomes keyframe.
        RETURN_FALSE_IF_FALSE(MakeCurrentFrameKeyframe());
    }

    // Record package data.
    if (options().kRecordBinaryLog) {
        logger().RecordPackage(kFrontendMonoLogIndex, reinterpret_cast<const char *>(&log_package_data_));
    }

    return true;
}

// Draw tracking results.
void FrontendMono::DrawTrackingResults(const std::string title) {
    Visualizor::ShowImageWithTrackedFeaturesWithId(
        title,
        ref_pyramid_left_->GetImage(0),
        cur_pyramid_left_->GetImage(0),
        *ref_pixel_uv_left_, *cur_pixel_uv_left_,
        *ref_ids_, *cur_ids_, tracked_status_,
        static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
        *ref_tracked_cnt_, *cur_vel_);
    Visualizor::WaitKey(1);
}

// Support for log recording.
void FrontendMono::RegisterLogPackages() {
    using namespace SLAM_DATA_LOG;

    std::unique_ptr<PackageInfo> package_ptr = std::make_unique<PackageInfo>();
    package_ptr->id = kFrontendMonoLogIndex;
    package_ptr->name = "frontend_mono";
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_keyframe"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_old_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_tracked_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_inliers"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_inliers_after_filter"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_new_features"});

    if (!logger_.RegisterPackage(package_ptr)) {
        ReportError("Failed to register package.");
    }
}

}
