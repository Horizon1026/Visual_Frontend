#include "frontend.h"
#include "pinhole.h"
#include "log_api.h"

#include "iostream"
#include "dirent.h"
#include "vector"
#include "cstring"

#include "opencv2/opencv.hpp"

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

int main() {
    LogInfo(YELLOW ">> Test visual frontend on euroc dataset." RESET_COLOR);

    std::vector<std::string> cam0_filenames;
    GetFilesInPath("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam0/data", cam0_filenames);
    std::sort(cam0_filenames.begin(), cam0_filenames.end());

    // Config frontend.
    cv::Mat image = cv::imread(cam0_filenames.front());
    VISUAL_FRONTEND::Frontend frontend(image.rows, image.cols);
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 50;

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
    frontend.feature_detector()->options().kMethod = FEATURE_DETECTOR::FeatureDetector::HARRIS;
    frontend.feature_detector()->options().kMinValidResponse = 20.0f;
    frontend.feature_detector()->options().kMinFeatureDistance = 25;

    // Config optical flow tracker.
    frontend.feature_tracker()->options().kMethod = OPTICAL_FLOW::LkMethod::LK_FAST;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 12;
    frontend.feature_tracker()->options().kPatchColHalfSize = 12;
    frontend.feature_tracker()->options().kMaxIteration = 10;
    frontend.feature_tracker()->options().kMaxConvergeResidual = 0.1f;

    // Config epipolar solver.
    frontend.epipolar_solver()->options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::EPIPOLAR_RANSAC;
    frontend.epipolar_solver()->options().kMaxEpipolarResidual = 2e-2f;

    for (const auto &filename : cam0_filenames) {
        cv::Mat cv_image = cv::imread(filename, 0);
        Image image;
        image.SetImage(cv_image.data, cv_image.rows, cv_image.cols);
        frontend.RunOnce(image);
    }


    return 0;
}
