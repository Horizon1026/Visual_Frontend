#include "frontend_mono.h"
#include "slam_operations.h"
#include "log_api.h"

// Debug.
#include "opencv2/opencv.hpp"

namespace VISUAL_FRONTEND {

// Debug.
void DrawReferenceResultsPrediction(const std::string title,
                                    const Image &ref_image,
                                    const Image &cur_image,
                                    const std::vector<Vec2> &ref_points,
                                    const std::vector<Vec2> &cur_points,
                                    const std::vector<uint32_t> &ref_ids,
                                    const std::vector<uint32_t> &cur_ids,
                                    const std::vector<Vec2> &ref_vel) {
    cv::Mat cv_ref_image(ref_image.rows(), ref_image.cols(), CV_8UC1, ref_image.data());
    cv::Mat cv_cur_image(cur_image.rows(), cur_image.cols(), CV_8UC1, cur_image.data());

    // Merge three images.
    cv::Mat merged_image(cv_cur_image.rows, cv_cur_image.cols * 2, CV_8UC1);
    for (int32_t v = 0; v < merged_image.rows; ++v) {
        for (int32_t u = 0; u < merged_image.cols; ++u) {
            if (u < cv_ref_image.cols) {
                merged_image.at<uchar>(v, u) = cv_ref_image.at<uchar>(v, u);
            } else {
                merged_image.at<uchar>(v, u) = cv_cur_image.at<uchar>(v, u - cv_cur_image.cols);
            }
        }
    }

    // Construct image to show.
    cv::Mat show_image(merged_image.rows, merged_image.cols, CV_8UC3);
    cv::cvtColor(merged_image, show_image, cv::COLOR_GRAY2BGR);

    // [left] Draw reference points.
    for (uint32_t i = 0; i < ref_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), 1, cv::Scalar(0, 0, 255), 3);
        cv::putText(show_image, std::to_string(ref_ids[i]), cv::Point2f(ref_points[i].x(), ref_points[i].y()),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 0, 255));
    }

    // [right] Draw result points.
    for (uint32_t i = 0; i < cur_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(cur_points[i].x() + cv_cur_image.cols, cur_points[i].y()), 1, cv::Scalar(255, 255, 0), 3);
        cv::putText(show_image, std::to_string(cur_ids[i]), cv::Point2f(cur_points[i].x() + cv_cur_image.cols, cur_points[i].y()),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 255, 0));
    }

    cv::imshow(title, show_image);
    cv::waitKey(0);
}

bool FrontendMono::RunOnce(const Image &cur_image) {
    if (cur_image.data() == nullptr) {
        return false;
    }

    if (camera_model_ == nullptr) {
        LogError("Camera model is nullptr, please set it to be pinhole or fisheye.");
        return false;
    }

    // Image process.
    std::copy_n(cur_image.data(), cur_image.rows() * cur_image.cols(), cur_pyramid_left_->GetImage(0).data());
    cur_pyramid_left_->CreateImagePyramid(4);
    LogInfo("---------------------------------------------------------");

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

    // Debug.
    DrawReferenceResultsPrediction("Tracking result",
                                   ref_pyramid_left_->GetImage(0),
                                   cur_pyramid_left_->GetImage(0),
                                   *ref_pixel_uv_left_,
                                   *cur_pixel_uv_left_,
                                   *ref_ids_,
                                   *cur_ids_,
                                   *ref_vel_);

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe_) {
        // Adjust result.
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_pixel_uv_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_ids_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_norm_xy_left_);
        SlamOperation::ReduceVectorByStatus(tracked_status_, *cur_vel_);
        tracked_status_.resize(cur_pixel_uv_left_->size(), static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED));

        // Detect new features in cur.
        feature_detector_->DetectGoodFeatures(cur_pyramid_left_->GetImage(0),
                                              options_.kMaxStoredFeaturePointsNumber,
                                              *cur_pixel_uv_left_);
        const uint32_t new_features_num = cur_pixel_uv_left_->size() - cur_ids_->size();
        for (uint32_t i = 0; i < new_features_num; ++i) {
            cur_ids_->emplace_back(feature_id_cnt_);
            cur_norm_xy_left_->emplace_back(Vec2::Zero());
            ++feature_id_cnt_;
        }

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
