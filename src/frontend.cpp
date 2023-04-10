#include "frontend.h"
#include "slam_memory.h"
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
                                    const std::vector<Vec2> &ref_vel) {
    cv::Mat cv_ref_image(ref_image.rows(), ref_image.cols(), CV_8UC1, ref_image.data());
    cv::Mat cv_cur_image(cur_image.rows(), cur_image.cols(), CV_8UC1, cur_image.data());

    // Merge three images.
    cv::Mat merged_image(cv_cur_image.rows * 2, cv_cur_image.cols * 2, CV_8UC1);
    for (int32_t v = 0; v < merged_image.rows; ++v) {
        for (int32_t u = 0; u < merged_image.cols; ++u) {
            if (v < cv_ref_image.rows && u < cv_ref_image.cols) {
                merged_image.at<uchar>(v, u) = cv_ref_image.at<uchar>(v, u);
            } else if (v < cv_ref_image.rows && u < cv_ref_image.cols * 2) {
                merged_image.at<uchar>(v, u) = cv_cur_image.at<uchar>(v, u - cv_cur_image.cols);
            } else if (v < cv_ref_image.rows * 2 && u < cv_ref_image.cols) {
                merged_image.at<uchar>(v, u) = cv_cur_image.at<uchar>(v - cv_cur_image.rows, u);
            }
        }
    }

    // Construct image to show.
    cv::Mat show_image(merged_image.rows, merged_image.cols, CV_8UC3);
    cv::cvtColor(merged_image, show_image, cv::COLOR_GRAY2BGR);

    // [Top left] Draw reference points.
    for (uint32_t i = 0; i < ref_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), 2, cv::Scalar(0, 0, 255), 3);
    }

    // [Top right] Draw result points.
    for (uint32_t i = 0; i < ref_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(cur_points[i].x() + cv_cur_image.cols, cur_points[i].y()), 2, cv::Scalar(255, 0, 0), 3);
    }

    // [Bottom left] Draw prediction points.
    for (uint32_t i = 0; i < ref_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(ref_points[i].x() + ref_vel[i].x(), ref_points[i].y() + ref_vel[i].y() + cv_cur_image.rows), 2, cv::Scalar(0, 255, 0), 3);
    }

    cv::imshow(title, show_image);
    cv::waitKey(0);
}

Frontend::Frontend(const uint32_t image_rows, const uint32_t image_cols) {
    const uint32_t size = image_rows * image_cols;
    stored_buff_ = (uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * size * 8);

    // Allocate memory for stored buffer.
    uint8_t *buf = stored_buff_;
    for (uint32_t i = 0; i < 4; ++i) {
        stored_pyramids_[i].SetRawImage(buf, image_rows, image_cols);
        buf += size;
        stored_pyramids_[i].SetPyramidBuff(buf);
        buf += size;
    }

}

Frontend::~Frontend() {
    SlamMemory::Free(stored_buff_);
}

bool Frontend::RunOnce(const Image &cur_image) {
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

    // Track features if ref frame is ok.
    if (ref_points_->size() != 0) {
        // Predict pixel location on current image by optical flow velocity.
        *cur_points_ = *ref_points_;    // Deep copy.
        // for (uint32_t i = 0; i < ref_vel_->size(); ++i) {
        //     (*cur_points_)[i] += (*ref_vel_)[i];
        // }

        // Track features from ref pyramid to cur pyramid.
        *cur_ids_ = *ref_ids_;
        tracked_status_.clear();
        feature_tracker_->TrackMultipleLevel(*ref_pyramid_left_, *cur_pyramid_left_, *ref_points_, *cur_points_, tracked_status_);

        // Compute optical flow velocity. It is useful for feature prediction.
        int32_t cnt = 0;
        cur_vel_->resize(ref_points_->size());
        for (uint32_t i = 0; i < ref_points_->size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                (*cur_vel_)[i] = (*cur_points_)[i] - (*ref_points_)[i];
                ++cnt;
            } else {
                (*cur_vel_)[i].setZero();
            }
        }
        LogInfo("Feature tracked " << ref_points_->size() << " / " << cnt << ".");

        // Reject outliers by essential/fundemantal matrix.
        cur_norm_xy_->resize(cur_points_->size());
        for (uint32_t i = 0; i < cur_points_->size(); ++i) {
            camera_model_->LiftToNormalizedPlaneAndUndistort((*cur_points_)[i], (*cur_norm_xy_)[i]);
        }
        Mat3 essential;
        epipolar_solver_->EstimateEssential(*ref_norm_xy_, *cur_norm_xy_, essential, tracked_status_);

        // Reject outliers' optical flow velocity. It means do not predict them at next tracking.
        cnt = 0;
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            if (tracked_status_[i] != static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                (*cur_vel_)[i].setZero();
            } else {
                ++cnt;
            }
        }
        *ref_vel_ = *cur_vel_;
        LogInfo("Essential reject outliers " << ref_norm_xy_->size() << " / " << cnt << ".");

        // Grid filter to make points sparsely.
        MatInt grid;
        int32_t grid_rows = 15;
        int32_t grid_cols = 15;
        float grid_row_step = cur_image.rows() / (grid_rows - 1);
        float grid_col_step = cur_image.cols() / (grid_cols - 1);
        grid.setZero(grid_rows, grid_cols);
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            const int32_t row = static_cast<int32_t>((*cur_points_)[i].y() / grid_row_step);
            const int32_t col = static_cast<int32_t>((*cur_points_)[i].x() / grid_col_step);
            if (grid(row, col) && tracked_status_[i] == static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                tracked_status_[i] = static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::NOT_TRACKED);
            } else {
                grid(row, col) = 1;
            }
        }
        cnt = 0;
        for (uint32_t i = 0; i < tracked_status_.size(); ++i) {
            if (tracked_status_[i] == static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED)) {
                ++cnt;
            }
        }
        LogInfo("Grid filter reject outliers " << ref_norm_xy_->size() << " / " << cnt << ".");

        // Adjust result.
        AdjustVectorByStatus(tracked_status_, *cur_points_);
        AdjustVectorByStatus(tracked_status_, *cur_ids_);
        AdjustVectorByStatus(tracked_status_, *cur_norm_xy_);
        AdjustVectorByStatus(tracked_status_, *cur_vel_);
        tracked_status_.resize(cur_points_->size(), static_cast<uint8_t>(OPTICAL_FLOW::TrackStatus::TRACKED));
    }

    // Debug.
    DrawReferenceResultsPrediction("reference - result - predict - none",
                                   ref_pyramid_left_->GetImage(0),
                                   cur_pyramid_left_->GetImage(0),
                                   *ref_points_,
                                   *cur_points_,
                                   *ref_vel_);

    // Check if cur_pyarmid should be keyframe.
    is_cur_image_keyframe_ = cur_points_->size() < options_.kMinDetectedFeaturePointsNumberInCurrentImage
                          || !options_.kSelfSelectKeyframe;

    // If frontend is configured to select keyframe by itself, frontend will track features from fixed keyframe to current frame.
    if (is_cur_image_keyframe_) {
        // Detect new features in cur.
        feature_detector_->DetectGoodFeatures(cur_pyramid_left_->GetImage(0),
                                              options_.kMaxStoredFeaturePointsNumber,
                                              *cur_points_);
        const uint32_t new_features_num = cur_points_->size() - cur_ids_->size();
        for (uint32_t i = 0; i < new_features_num; ++i) {
            cur_ids_->emplace_back(feature_id_cnt_);
            cur_norm_xy_->emplace_back(Vec2::Zero());
            ++feature_id_cnt_;
        }

        // When current frame becomes keyframe, prediction will not work in next tracking.
        cur_vel_->clear();

        // Replace ref with cur.
        ExchangePointer(&ref_pyramid_left_, &cur_pyramid_left_);
        ExchangePointer(&ref_points_, &cur_points_);
        ExchangePointer(&ref_ids_, &cur_ids_);
        ExchangePointer(&ref_norm_xy_, &cur_norm_xy_);
        ExchangePointer(&ref_vel_, &cur_vel_);
    }

    return true;
}

bool Frontend::RunOnce(const Image &image_left, const Image &image_right) {
    if (image_left.data() == nullptr || image_right.data() == nullptr) {
        return false;
    }

    if (camera_model_ == nullptr) {
        LogError("Camera model is nullptr, please set it to be pinhole or fisheye.");
        return false;
    }

    // TODO:

    return true;
}

template<typename T>
void Frontend::ExchangePointer(T **ptr1, T** ptr2) {
    T *ptr_tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = ptr_tmp;
}

template<typename T, typename StatusType>
void Frontend::AdjustVectorByStatus(const std::vector<StatusType> &status,
                                    std::vector<T> &v)  {
    uint32_t j = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        if (status[i] == static_cast<StatusType>(1)) {
            v[j] = v[i];
            ++j;
        }
    }
    v.resize(j);
}

}
