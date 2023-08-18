#include "frontend_mono.h"
#include "frontend_stereo.h"
#include "pinhole.h"
#include "optical_flow_basic_klt.h"
#include "optical_flow_affine_klt.h"
#include "optical_flow_lssd_klt.h"
#include "census.h"
#include "log_report.h"

#include "visualizor.h"

#include "iostream"
#include "dirent.h"
#include "vector"
#include "cstring"

namespace {
    using FeatureType = FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::FastFeature>;
    using KltType = FEATURE_TRACKER::OpticalFlowBasicKlt;
}

bool GetFilesInPath(std::string dir, std::vector<std::string> &filenames) {
    DIR *ptr_dir;
    struct dirent *ptr;
    if (!(ptr_dir = opendir(dir.c_str()))) {
        ReportError("Cannot open dir " << dir);
        return false;
    }

    filenames.reserve(1000);

    while ((ptr = readdir(ptr_dir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            filenames.emplace_back(dir + "/" + ptr->d_name);
        }
    }

    closedir(ptr_dir);

    return true;
}

void TestFrontendMono(const std::vector<std::string> &cam0_filenames) {
    ReportInfo(">> Test frontend mono.");

    // Config frontend.
    GrayImage image;
    Visualizor::LoadImage(cam0_filenames.front(), image);
    VISUAL_FRONTEND::FrontendMono frontend(image.rows(), image.cols());
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 40;

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
    Vec5 distort_param;
    distort_param << k1, k2, k3, p1, p2;
    frontend.camera_model()->SetDistortionParameter(distort_param);

    // Config image processor.
    // frontend.image_processor() = std::make_unique<IMAGE_PROCESSOR::CensusProcessor>();

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 35;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 10;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 10;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<KltType>();
    frontend.feature_tracker()->options().kMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 6;
    frontend.feature_tracker()->options().kPatchColHalfSize = 6;
    frontend.feature_tracker()->options().kMaxIteration = 15;

    // Config epipolar solver.
    // frontend.epipolar_solver() = std::make_unique<VISION_GEOMETRY::EpipolarSolver>();
    // frontend.epipolar_solver()->options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::kRansac;
    // frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    for (const auto &filename : cam0_filenames) {
        GrayImage image;
        Visualizor::LoadImage(filename, image);
        frontend.RunOnce(image);
    }
}

void TestFrontendStereo(const std::vector<std::string> &cam0_filenames, const std::vector<std::string> &cam1_filenames) {
    ReportInfo(">> Test frontend stereo.");

    // Config frontend.
    GrayImage image;
    Visualizor::LoadImage(cam0_filenames.front(), image);
    VISUAL_FRONTEND::FrontendStereo frontend(image.rows(), image.cols());
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 30;

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
    Vec5 distort_param;
    distort_param << k1, k2, k3, p1, p2;
    frontend.camera_model()->SetDistortionParameter(distort_param);

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 35;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 10;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 10;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<KltType>();
    frontend.feature_tracker()->options().kMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 6;
    frontend.feature_tracker()->options().kPatchColHalfSize = 6;
    frontend.feature_tracker()->options().kMaxIteration = 15;

    // Config epipolar solver.
    frontend.epipolar_solver() = std::make_unique<VISION_GEOMETRY::EpipolarSolver>();
    frontend.epipolar_solver()->options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::kRansac;
    frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    for (unsigned i = 0; i < cam0_filenames.size(); ++i) {
        GrayImage image_left;
        GrayImage image_right;
        Visualizor::LoadImage(cam0_filenames[i], image_left);
        Visualizor::LoadImage(cam1_filenames[i], image_right);
        frontend.RunOnce(image_left, image_right);
    }
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test visual frontend on euroc dataset." RESET_COLOR);

    std::vector<std::string> cam0_filenames;
    if (!GetFilesInPath("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam0/data", cam0_filenames)) {
        RETURN_FALSE_IF_FALSE(GetFilesInPath("D:/My_Github/Datasets/MH_05_difficult/mav0/cam0/data", cam0_filenames));
    }
    std::sort(cam0_filenames.begin(), cam0_filenames.end());

    std::vector<std::string> cam1_filenames;
    if (!GetFilesInPath("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam1/data", cam1_filenames)) {
        RETURN_FALSE_IF_FALSE(GetFilesInPath("D:/My_Github/Datasets/MH_05_difficult/mav0/cam1/data", cam1_filenames));
    }
    std::sort(cam1_filenames.begin(), cam1_filenames.end());

    if (argc > 2) {
        const int32_t steps_cnt = atoi(argv[2]);
        ReportInfo("Only run " << steps_cnt << " steps, full steps is " << cam0_filenames.size() << ".");
        cam0_filenames.resize(steps_cnt);
        cam1_filenames.resize(steps_cnt);
    }

    if (argc > 1 && std::string(argv[1]) == "stereo") {
        ReportInfo("Run stereo visual frontend.");
        TestFrontendStereo(cam0_filenames, cam1_filenames);
    } else if (argc > 1 && std::string(argv[1]) == "mono") {
        ReportInfo("Run mono visual frontend.");
        TestFrontendMono(cam0_filenames);
    } else {
        ReportInfo("Please config 'mono' or 'stereo' to run visual frontend. Default mono.");
        TestFrontendMono(cam0_filenames);
    }

    return 0;
}
