#include "frontend_mono.h"
#include "pinhole.h"
#include "optical_flow_lk.h"
#include "optical_flow_klt.h"
#include "log_api.h"

#include "iostream"
#include "dirent.h"
#include "vector"
#include "cstring"

#include "opencv2/opencv.hpp"

void DrawReferenceResults(const std::string title,
                          const Image &ref_image,
                          const Image &cur_image,
                          const std::vector<Vec2> &ref_points,
                          const std::vector<Vec2> &cur_points,
                          const std::vector<uint32_t> &ref_ids,
                          const std::vector<uint32_t> &cur_ids,
                          const std::vector<uint32_t> &ref_tracked_cnt,
                          const std::vector<Vec2> &ref_vel) {
    cv::Mat cv_ref_image(ref_image.rows(), ref_image.cols(), CV_8UC1, ref_image.data());
    cv::Mat cv_cur_image(cur_image.rows(), cur_image.cols(), CV_8UC1, cur_image.data());

    // Merge images.
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
        int32_t color = 255 - ref_tracked_cnt[i] * 50;
        if (color < 0) {
            color = 0;
        }
        cv::circle(show_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), 1, cv::Scalar(0, 255 - color, 255), 3);
        cv::putText(show_image, std::to_string(ref_ids[i]), cv::Point2f(ref_points[i].x(), ref_points[i].y()),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 255 - color, 255));
    }

    // [right] Draw result points.
    for (uint32_t i = 0; i < cur_points.size(); ++i) {
        cv::circle(show_image, cv::Point2f(cur_points[i].x() + cv_cur_image.cols, cur_points[i].y()), 1, cv::Scalar(255, 255, 0), 3);
        cv::putText(show_image, std::to_string(cur_ids[i]), cv::Point2f(cur_points[i].x() + cv_cur_image.cols, cur_points[i].y()),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 255, 0));
    }

    cv::imshow(title, show_image);
    cv::waitKey(0);
}

void GetFilesInPath(std::string dir, std::vector<std::string> &filenames) {
    DIR *ptr_dir;
    struct dirent *ptr;
    if (!(ptr_dir = opendir(dir.c_str()))) {
        LogError("Cannot open dir " << dir);
        return;
    }

    filenames.reserve(1000);

    while ((ptr = readdir(ptr_dir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            filenames.emplace_back(dir + "/" + ptr->d_name);
        }
    }

    closedir(ptr_dir);
}

void TestFrontendMono(const std::vector<std::string> &cam0_filenames) {
    LogInfo(">> Test frontend mono.");

    // Config frontend.
    cv::Mat image = cv::imread(cam0_filenames.front());
    VISUAL_FRONTEND::FrontendMono frontend(image.rows, image.cols);
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 70;
    frontend.VisualizeResult = DrawReferenceResults;

    // Config camera model.
    const float fx = 458.654f;
    const float fy = 457.296f;
    const float cx = 752.0f / 2.0f;
    const float cy = 240.0f;
    const float k1 = -0.28340811f;
    const float k2 = 0.07395907f;
    const float k3 = 0.0f;
    const float p1 = 0.00019359f;
    const float p2 = 1.76187114e-05f;
    frontend.camera_model() = std::make_unique<SENSOR_MODEL::Pinhole>();
    frontend.camera_model()->SetIntrinsicParameter(fx, fy, cx, cy);
    frontend.camera_model()->SetDistortionParameter(Vec5(k1, k2, k3, p1, p2));

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FEATURE_DETECTOR::FeatureDetector>();
    frontend.feature_detector()->options().kMethod = FEATURE_DETECTOR::FeatureDetector::HARRIS;
    frontend.feature_detector()->options().kMinValidResponse = 20.0f;
    frontend.feature_detector()->options().kMinFeatureDistance = 25;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 12;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 12;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<OPTICAL_FLOW::OpticalFlowLk>();
    frontend.feature_tracker()->options().kMethod = OPTICAL_FLOW::Method::LK_FAST;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 10;
    frontend.feature_tracker()->options().kPatchColHalfSize = 10;
    frontend.feature_tracker()->options().kMaxIteration = 15;
    frontend.feature_tracker()->options().kMaxConvergeResidual = 1.0f;

    // Config epipolar solver.
    frontend.epipolar_solver() = std::make_unique<VISION_GEOMETRY::EpipolarSolver>();
    frontend.epipolar_solver()->options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::EPIPOLAR_RANSAC;
    frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    for (const auto &filename : cam0_filenames) {
        cv::Mat cv_image = cv::imread(filename, 0);
        Image image;
        image.SetImage(cv_image.data, cv_image.rows, cv_image.cols);
        frontend.RunOnce(image);
    }
}

int main() {
    LogInfo(YELLOW ">> Test visual frontend on euroc dataset." RESET_COLOR);

    std::vector<std::string> cam0_filenames;
    GetFilesInPath("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam0/data", cam0_filenames);
    std::sort(cam0_filenames.begin(), cam0_filenames.end());

    std::vector<std::string> cam1_filenames;
    GetFilesInPath("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam1/data", cam1_filenames);
    std::sort(cam1_filenames.begin(), cam1_filenames.end());

    TestFrontendMono(cam0_filenames);

    return 0;
}
