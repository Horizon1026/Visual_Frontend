#include "frontend_mono.h"
#include "slam_operations.h"
#include "log_report.h"
#include "tick_tock.h"
#include "visualizor.h"

namespace VISUAL_FRONTEND {

constexpr uint32_t kFrontendMonoLogIndex = 1;

bool FrontendMono::ProcessSourceImage(const GrayImage &cur_image) {
    RETURN_FALSE_IF(cur_image.data() == nullptr);

    if (image_processor() == nullptr) {
        std::copy_n(cur_image.data(), cur_image.rows() * cur_image.cols(), cur_pyramid_left()->GetImage(0).data());
    } else {
        image_processor()->Process(cur_image, cur_pyramid_left()->GetImage(0));
    }

    cur_pyramid_left()->CreateImagePyramid(4);
    return true;
}

bool FrontendMono::PredictPixelLocation() {
    *cur_pixel_uv_left() = *ref_pixel_uv_left();    // Deep copy.
    for (uint32_t i = 0; i < ref_vel()->size(); ++i) {
        (*cur_pixel_uv_left())[i] += (*ref_vel())[i];
    }
    return true;
}

bool FrontendMono::TrackFeatures() {
    *cur_ids() = *ref_ids();
    tracked_status().clear();
    if (!feature_tracker()->TrackFeatures(*ref_pyramid_left(), *cur_pyramid_left(), *ref_pixel_uv_left(), *cur_pixel_uv_left(), tracked_status())) {
        ReportError("feature_tracker()->TrackFeatures error.");
        return false;
    }

    // Record log data.
    log_package_data_.num_of_old_features = tracked_status().size();
    log_package_data_.num_of_tracked_features = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem() {
    cur_norm_xy_left()->resize(cur_pixel_uv_left()->size());
    for (uint32_t i = 0; i < cur_pixel_uv_left()->size(); ++i) {
        camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left())[i], (*cur_norm_xy_left())[i]);
    }

    return true;
}

bool FrontendMono::RejectOutliersByEpipolarConstrain() {
    Mat3 essential;
    if (!epipolar_solver()->EstimateEssential(*ref_norm_xy_left(), *cur_norm_xy_left(), essential, tracked_status())) {
        ReportError("epipolar_solver()->EstimateEssential error");
        return false;
    }

    // Record log data.
    log_package_data_.num_of_inliers = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::RejectOutliersByTrackingBack() {
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

    // Record log data.
    log_package_data_.num_of_inliers = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::ComputeOpticalFlowVelocity() {
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

bool FrontendMono::SparsifyTrackedFeatures() {
    feature_detector()->SparsifyFeatures(*cur_pixel_uv_left(),
                                         cur_pyramid_left()->GetImage(0).rows(),
                                         cur_pyramid_left()->GetImage(0).cols(),
                                         static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
                                         static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kNotTracked),
                                         tracked_status());

    // Record log data.
    log_package_data_.num_of_inliers_after_filter = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    return true;
}

bool FrontendMono::SelectKeyframe() {
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    is_cur_image_keyframe() = tracked_num < options().kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options().kSelfSelectKeyframe;

    // Record log data.
    log_package_data_.is_keyframe = static_cast<uint8_t>(is_cur_image_keyframe());
    return true;
}

bool FrontendMono::AdjustTrackingResultByStatus() {
    // Update tracked statis result.
    for (uint32_t i = 0; i < tracked_status().size(); ++i) {
        if (tracked_status()[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            ++(*ref_tracked_cnt())[i];
        }
    }

    // Adjust result.
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_pixel_uv_left());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_ids());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_norm_xy_left());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *cur_vel());
    SlamOperation::ReduceVectorByStatus(tracked_status(), *ref_tracked_cnt());
    tracked_status().resize(cur_pixel_uv_left()->size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

    return true;
}

bool FrontendMono::SupplementNewFeatures(const GrayImage &cur_image_left) {
    feature_detector()->DetectGoodFeatures(cur_image_left,
                                           options().kMaxStoredFeaturePointsNumber,
                                           *cur_pixel_uv_left());

    const uint32_t new_features_num = cur_pixel_uv_left()->size() - cur_ids()->size();

    for (uint32_t i = 0; i < new_features_num; ++i) {
        cur_ids()->emplace_back(feature_id_cnt());
        cur_norm_xy_left()->emplace_back(Vec2::Zero());
        ref_tracked_cnt()->emplace_back(1);
        ++feature_id_cnt();
    }

    // Record log data.
    log_package_data_.num_of_new_features = new_features_num;
    return true;
}

bool FrontendMono::MakeCurrentFrameKeyframe() {
    // When current frame becomes keyframe, prediction will not work in next tracking.
    cur_vel()->clear();

    // Replace ref with cur.
    SlamOperation::ExchangePointer(&ref_pyramid_left(), &cur_pyramid_left());
    SlamOperation::ExchangePointer(&ref_pixel_uv_left(), &cur_pixel_uv_left());
    SlamOperation::ExchangePointer(&ref_ids(), &cur_ids());
    SlamOperation::ExchangePointer(&ref_norm_xy_left(), &cur_norm_xy_left());
    SlamOperation::ExchangePointer(&ref_vel(), &cur_vel());

    return true;
}

bool FrontendMono::RunOnce(const GrayImage &cur_image) {
    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());
    // GrayImage process.
    RETURN_FALSE_IF_FALSE(ProcessSourceImage(cur_image));

    // Track features if ref frame is ok.
    if (!ref_pixel_uv_left()->empty()) {
        // Predict pixel location on current image by optical flow velocity.
        RETURN_FALSE_IF_FALSE(PredictPixelLocation());
        // Track features from ref pyramid to cur pyramid.
        RETURN_FALSE_IF_FALSE(TrackFeatures());
        // Lift and do undistortion.
        RETURN_FALSE_IF_FALSE(LiftAllPointsFromPixelToNormalizedPlaneAndUndistortThem());

        if (epipolar_solver() == nullptr) {
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
    if (is_cur_image_keyframe()) {
        // Prepare to make current frame to be reference frame (keyframe).
        RETURN_FALSE_IF_FALSE(AdjustTrackingResultByStatus());
        // Detect new features in cur.
        RETURN_FALSE_IF_FALSE(SupplementNewFeatures(cur_image));
        // Current frame becomes keyframe.
        RETURN_FALSE_IF_FALSE(MakeCurrentFrameKeyframe());
    }

    // Record package data.
    if (options().kEnableRecordBinaryLog) {
        logger().RecordPackage(kFrontendMonoLogIndex, reinterpret_cast<const char *>(&log_package_data_));
    }

    return true;
}

// Draw tracking results.
void FrontendMono::DrawTrackingResults(const std::string title) {
    if (!options().kEnableVisualizeResult) {
        return;
    }

    Visualizor::ShowImageWithTrackedFeaturesWithId(
        title,
        ref_pyramid_left()->GetImage(0),
        cur_pyramid_left()->GetImage(0),
        *ref_pixel_uv_left(), *cur_pixel_uv_left(),
        *ref_ids(), *cur_ids(), tracked_status(),
        static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
        *ref_tracked_cnt(), *cur_vel());
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

    if (!logger().RegisterPackage(package_ptr)) {
        ReportError("Failed to register package.");
    }
}

// Update frontend result.
void FrontendMono::UpdateFrontendOutputData() {
    output_data().features_id.clear();
    output_data().observes_per_frame.clear();
    output_data().optical_velocity_in_ref_view.clear();
}

}
