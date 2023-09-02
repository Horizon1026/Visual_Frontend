#include "frontend_stereo.h"
#include "slam_operations.h"
#include "log_report.h"
#include "tick_tock.h"
#include "visualizor.h"

namespace VISUAL_FRONTEND {

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

bool FrontendStereo::RunOnce(const GrayImage &cur_image_left, const GrayImage &cur_image_right) {
    ReportInfo("---------------------------------------------------------");

    // If components is not valid, return false.
    RETURN_FALSE_IF_FALSE(CheckAllComponents());

    // GrayImage process.
    RETURN_FALSE_IF_FALSE(ProcessSourceImage(cur_image_left, cur_image_right));

    // Track features if ref frame is ok.
    if (ref_pixel_uv_left()->size() != 0) {
        // Predict pixel location on current image by optical flow velocity.
        *cur_pixel_uv_left() = *ref_pixel_uv_left();
        for (uint32_t i = 0; i < ref_vel()->size(); ++i) {
            (*cur_pixel_uv_left())[i] += (*ref_vel())[i];
        }

        // Track features from ref pyramid to cur pyramid.
        *cur_ids() = *ref_ids();
        tracked_status().clear();
        if (!feature_tracker()->TrackFeatures(*ref_pyramid_left(), *cur_pyramid_left(), *ref_pixel_uv_left(), *cur_pixel_uv_left(), tracked_status())) {
            ReportError("feature_tracker()->TrackFeatures track from ref_left to cur_left error.");
            return false;
        }
        ReportInfo("After optical flow tracking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status().size());

        // Reject outliers by essential/fundemantal matrix.
        Mat3 essential;
        cur_norm_xy_left()->resize(cur_pixel_uv_left()->size());
        for (uint32_t i = 0; i < cur_pixel_uv_left()->size(); ++i) {
            camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_left())[i], (*cur_norm_xy_left())[i]);
        }
        if (!epipolar_solver()->EstimateEssential(*ref_norm_xy_left(), *cur_norm_xy_left(), essential, tracked_status())) {
            ReportError("epipolar_solver()->EstimateEssential error");
            return false;
        }
        ReportInfo("After essential checking, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status().size());

        // Compute optical flow velocity. It is useful for feature prediction.
        cur_vel()->resize(ref_pixel_uv_left()->size());
        for (uint32_t i = 0; i < ref_pixel_uv_left()->size(); ++i) {
            if (tracked_status()[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
                (*cur_vel())[i] = (*cur_pixel_uv_left())[i] - (*ref_pixel_uv_left())[i];
            } else {
                (*cur_vel())[i].setZero();
            }
        }
        *ref_vel() = *cur_vel();

        // Track features from cur pyramid left to cur pyramid right.
        // ReportInfo("cur_image_right " << LogPtr(cur_image_right.data()));
        // ReportInfo("feature_detector() " << LogPtr(feature_detector().get()));
        detected_features_in_cur_right_.clear();
        feature_detector()->DetectGoodFeatures(cur_image_right,
                                               options().kMaxStoredFeaturePointsNumber,
                                               detected_features_in_cur_right_);
        // ReportInfo("detected_features_in_cur_right_ size " << detected_features_in_cur_right_.size());

        // ReportInfo("descriptor_ " << LogPtr(descriptor_.get()));
        descriptor_->options().kLength = 256;
        descriptor_->options().kHalfPatchSize = 16;
        cur_descriptor_left_.clear();
        cur_descriptor_right_.clear();
        // ReportInfo("cur_image_left " << LogPtr(cur_image_left.data()));
        // ReportInfo("*cur_pixel_uv_left() size " << cur_pixel_uv_left()->size());
        descriptor_->Compute(cur_image_left, *cur_pixel_uv_left(), cur_descriptor_left_);
        // ReportInfo("*cur_pixel_uv_left() size " << cur_pixel_uv_left()->size());
        ReportInfo("cur_descriptor_left_ size " << cur_descriptor_left_.size());
        descriptor_->Compute(cur_image_right, detected_features_in_cur_right_, cur_descriptor_right_);
        ReportInfo("cur_descriptor_right_ size " << cur_descriptor_right_.size());

        feature_matcher_->options().kMaxValidPredictRowDistance = 50;
        feature_matcher_->options().kMaxValidPredictColDistance = 100;
        feature_matcher_->options().kMaxValidDescriptorDistance = 60;

        cur_pixel_uv_right()->clear();
        tracked_status().clear();
        if (!feature_matcher_->NearbyMatch(cur_descriptor_left_, cur_descriptor_right_,
                                           *cur_pixel_uv_left(), detected_features_in_cur_right_,
                                           *cur_pixel_uv_right(), tracked_status())) {
            ReportError("feature_matcher_->NearbyMatch track from cur_left to cur_right error.");
            return false;
        }
        // ReportInfo("*cur_pixel_uv_right() size " << cur_pixel_uv_right()->size());
        ReportInfo("tracked_status() size " << tracked_status().size());

        // Reject outliers by essential/fundemantal matrix.
        cur_norm_xy_right()->resize(cur_pixel_uv_right()->size());
        for (uint32_t i = 0; i < cur_pixel_uv_right()->size(); ++i) {
            camera_model()->LiftToNormalizedPlaneAndUndistort((*cur_pixel_uv_right())[i], (*cur_norm_xy_right())[i]);
        }
        if (!epipolar_solver()->EstimateEssential(*cur_norm_xy_left(), *cur_norm_xy_right(), essential, tracked_status())) {
            ReportError("epipolar_solver()->EstimateEssential error");
            return false;
        }

        // Grid filter to make points sparsely.
        feature_detector()->SparsifyFeatures(*cur_pixel_uv_left(),
                                             cur_pyramid_left()->GetImage(0).rows(),
                                             cur_pyramid_left()->GetImage(0).cols(),
                                             static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked),
                                             static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kNotTracked),
                                             tracked_status());
        ReportInfo("After grid filtering, tracked / to_track is " << SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked))
            << " / " << tracked_status().size());
    }

    // Check if cur_pyarmid should be keyframe.
    const uint32_t tracked_num = SlamOperation::StatisItemInVector(tracked_status(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    is_cur_image_keyframe() = tracked_num < options().kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options().kSelfSelectKeyframe;
    if (is_cur_image_keyframe()) {
        ReportInfo("Current frame is keyframe.");
    }

    // Visualize result when this API is defined.
    DrawTrackingResults("Frontend Mono Tracking Result");

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe()) {
        // Prepare to make current frame to be reference frame (keyframe).
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
        tracked_status().resize(cur_pixel_uv_left()->size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));

        // Detect new features in cur.
        feature_detector()->DetectGoodFeatures(cur_image_left,
                                               options().kMaxStoredFeaturePointsNumber,
                                               *cur_pixel_uv_left());
        const uint32_t new_features_num = cur_pixel_uv_left()->size() - cur_ids()->size();
        for (uint32_t i = 0; i < new_features_num; ++i) {
            cur_ids()->emplace_back(feature_id_cnt());
            cur_norm_xy_left()->emplace_back(Vec2::Zero());
            cur_norm_xy_right()->emplace_back(Vec2::Zero());
            ref_tracked_cnt()->emplace_back(1);
            ++feature_id_cnt();
        }

        // Current frame becomes keyframe.
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
    }

    return true;
}

// Draw tracking results.
void FrontendStereo::DrawTrackingResults(const std::string title) {
    if (!options().kEnableVisualizeResult) {
        return;
    }

    Visualizor::ShowImageWithTrackedFeaturesWithId(
        title,
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
        *ref_tracked_cnt(),
        *cur_vel()
    );
    Visualizor::WaitKey(1);
}

}
