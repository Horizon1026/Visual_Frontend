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
#include "unistd.h"

namespace {
    using FeatureType = FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::FastFeature>;
    using KltType = FEATURE_TRACKER::OpticalFlowBasicKlt;

    constexpr bool kEnableDrawingOutputResult = true;
    constexpr bool kEnableRecordCurveLog = true;
    constexpr bool kEnableRecordImageLog = true;
    constexpr bool kEnableDrawingTrackingResult = false;
}

void ShowFrontendMonoOutput(const VISUAL_FRONTEND::FrontendMono &frontend,
                            const GrayImage &image) {
    // Show output data.
    if (kEnableDrawingOutputResult) {
        const auto &output = frontend.output_data();
        std::vector<Vec2> pixel_uv;
        pixel_uv.reserve(output.features_id.size());
        for (uint32_t i = 0; i < output.observes_per_frame.size(); ++i) {
            if (output.tracked_cnt[i] < 2) {
                break;
            }

            const auto &observe_per_view = output.observes_per_frame[i];
            pixel_uv.emplace_back(observe_per_view[0].raw_pixel_uv);
        }
        std::vector<uint8_t> status(pixel_uv.size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
        Visualizor::ShowImageWithTrackedFeatures(
            "Mono frontend output", image, pixel_uv, pixel_uv, status,
            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)
        );
        Visualizor::WaitKey(1);
    }
}

void TestFrontendMono(const std::vector<std::string> &cam0_filenames) {
    ReportInfo(">> Test frontend mono.");

    // Config frontend.
    GrayImage image;
    Visualizor::LoadImage(cam0_filenames.front(), image);
    VISUAL_FRONTEND::FrontendMono frontend(image.rows(), image.cols());
    frontend.options().kEnableRecordBinaryCurveLog = kEnableRecordCurveLog;
    frontend.options().kEnableRecordBinaryImageLog = kEnableRecordImageLog;
    frontend.options().kEnableShowVisualizeResult = kEnableDrawingTrackingResult;
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 40;
    frontend.Initialize("../output/frontend_mono_log.binlog");

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
    frontend.camera_model()->SetDistortionParameter(std::vector<float>{k1, k2, k3, p1, p2});

    // Config image processor.
    // frontend.image_processor() = std::make_unique<IMAGE_PROCESSOR::CensusProcessor>();

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 30;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 10;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 10;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<KltType>();
    frontend.feature_tracker()->options().kMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 6;
    frontend.feature_tracker()->options().kPatchColHalfSize = 6;
    frontend.feature_tracker()->options().kMaxIteration = 15;

    // Config epipolar solver. Disable it for better performance.
    // frontend.epipolar_solver() = std::make_unique<VISION_GEOMETRY::EpipolarSolver>();
    // frontend.epipolar_solver()->options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::kRansac;
    // frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    for (const auto &filename : cam0_filenames) {
        GrayImage image;
        Visualizor::LoadImage(filename, image);
        frontend.RunOnce(image);
        ShowFrontendMonoOutput(frontend, image);
    }
}

void ShowFrontendStereoOutput(const VISUAL_FRONTEND::FrontendStereo &frontend,
                              const GrayImage &image_left,
                              const GrayImage &image_right) {
    // Show output data.
    if (kEnableDrawingOutputResult) {
        const auto &output = frontend.output_data();
        std::vector<Vec2> pixel_uv_left;
        std::vector<Vec2> pixel_uv_right;
        std::vector<Vec2> rectify_uv_left;
        std::vector<Vec2> rectify_uv_right;
        std::vector<uint8_t> status;
        pixel_uv_left.reserve(output.features_id.size());
        pixel_uv_right.reserve(output.features_id.size());
        rectify_uv_left.reserve(output.features_id.size());
        rectify_uv_right.reserve(output.features_id.size());
        status.reserve(output.features_id.size());

        for (uint32_t i = 0; i < output.observes_per_frame.size(); ++i) {
            if (output.tracked_cnt[i] < 2) {
                break;
            }

            Vec2 pixel_uv = Vec2::Zero();

            const auto &observe_per_view = output.observes_per_frame[i];
            frontend.camera_model()->LiftFromNormalizedPlaneToImagePlane(observe_per_view[0].rectified_norm_xy, pixel_uv);
            rectify_uv_left.emplace_back(pixel_uv);
            pixel_uv_left.emplace_back(observe_per_view[0].raw_pixel_uv);

            if (observe_per_view.size() > 1) {
                frontend.camera_model()->LiftFromNormalizedPlaneToImagePlane(observe_per_view[1].rectified_norm_xy, pixel_uv);
                rectify_uv_right.emplace_back(pixel_uv);
                pixel_uv_right.emplace_back(observe_per_view[0].raw_pixel_uv);

                status.emplace_back(static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
            } else {
                rectify_uv_right.emplace_back(Vec2::Zero());
                pixel_uv_right.emplace_back(Vec2::Zero());
                status.emplace_back(static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kLargeResidual));
            }
        }

        MatImg mat_image_left, mat_image_right;
        mat_image_left.resize(image_left.rows(), image_left.cols());
        mat_image_right.resize(image_left.rows(), image_left.cols());
        GrayImage rectify_left(mat_image_left);
        GrayImage rectify_right(mat_image_right);
        frontend.camera_model()->CorrectDistortedImage(image_left, rectify_left);
        frontend.camera_model()->CorrectDistortedImage(image_right, rectify_right);

        Visualizor::ShowImageWithTrackedFeatures(
            "Stereo frontend rectify output", rectify_left, rectify_right, rectify_uv_left, rectify_uv_right, status,
            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
        Visualizor::ShowImageWithTrackedFeatures(
            "Stereo frontend raw output", image_left, image_right, pixel_uv_left, pixel_uv_right, status,
            static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
        Visualizor::WaitKey(0);
    }
}

void TestFrontendStereo(const std::vector<std::string> &cam0_filenames, const std::vector<std::string> &cam1_filenames) {
    ReportInfo(">> Test frontend stereo.");

    // Config frontend.
    GrayImage image;
    Visualizor::LoadImage(cam0_filenames.front(), image);
    VISUAL_FRONTEND::FrontendStereo frontend(image.rows(), image.cols());
    frontend.options().kEnableRecordBinaryCurveLog = kEnableRecordCurveLog;
    frontend.options().kEnableRecordBinaryImageLog = kEnableRecordImageLog;
    frontend.options().kEnableShowVisualizeResult = kEnableDrawingTrackingResult;
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 30;
    frontend.Initialize("../output/frontend_stereo_log.binlog");

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
    frontend.camera_model()->SetDistortionParameter(std::vector<float>{k1, k2, k3, p1, p2});

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 30;
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

    for (unsigned i = 0; i < cam0_filenames.size(); ++i) {
        GrayImage image_left;
        GrayImage image_right;
        Visualizor::LoadImage(cam0_filenames[i], image_left);
        Visualizor::LoadImage(cam1_filenames[i], image_right);
        frontend.RunOnce(image_left, image_right);
        ShowFrontendStereoOutput(frontend, image_left, image_right);
    }
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test visual frontend on euroc dataset." RESET_COLOR);

    std::vector<std::string> cam0_filenames;

    if (!SlamOperation::GetFilesNameInDirectory("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam0/data", cam0_filenames)) {
        RETURN_FALSE_IF_FALSE(SlamOperation::GetFilesNameInDirectory("D:/My_Github/Datasets/MH_01_easy/mav0/cam0/data", cam0_filenames));
    }
    std::sort(cam0_filenames.begin(), cam0_filenames.end());

    std::vector<std::string> cam1_filenames;
    if (!SlamOperation::GetFilesNameInDirectory("/home/horizon/Desktop/date_sets/euroc/MH_01_easy/mav0/cam1/data", cam1_filenames)) {
        RETURN_FALSE_IF_FALSE(SlamOperation::GetFilesNameInDirectory("D:/My_Github/Datasets/MH_01_easy/mav0/cam1/data", cam1_filenames));
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
