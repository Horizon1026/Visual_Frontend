#include "census.h"
#include "frontend_mono.h"
#include "frontend_stereo.h"
#include "optical_flow_affine_klt.h"
#include "optical_flow_basic_klt.h"
#include "optical_flow_lssd_klt.h"
#include "camera_model.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"

#include "image_painter.h"
#include "visualizor_2d.h"

#include "cstring"
#include "dirent.h"
#include "iostream"
#include "unistd.h"
#include "vector"

using namespace slam_visualizor;
using namespace image_painter;

namespace {
using FeatureType = feature_detector::FeaturePointFastDetector;
using KltType = feature_tracker::OpticalFlowBasicKlt;

constexpr bool kEnableDrawingOutputResult = true;
constexpr bool kEnableRecordCurveLog = true;
constexpr bool kEnableRecordImageLog = true;
constexpr bool kEnableDrawingTrackingResult = true;
}  // namespace

void ShowFrontendMonoOutput(const visual_frontend::FrontendMono &frontend, const GrayImage &image) {
    // Show output data.
    if (kEnableDrawingOutputResult) {
        const auto &output = frontend.output_data();

        // Camera name of each camera is camera_name[camera_id].
        std::vector<std::string> camera_name = {"left", "right"};

        if (image.data() != nullptr) {
            // Convert gray image to rgb image.
            RgbImage rgb_image;
            uint8_t *rgb_buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
            rgb_image.SetImage(rgb_buf, image.rows(), image.cols(), true);
            ImagePainter::ConvertUint8ToRgb(image.data(), rgb_image.data(), image.rows() * image.cols());

            // Draw all observed features in this frame and this camera image.
            for (uint32_t i = 0; i < output.features_id.size(); ++i) {
                const Vec2 pixel_uv = output.observes_per_frame[i][0].raw_pixel_uv;
                const RgbPixel pixel_color = RgbColor::kCyan;
                ImagePainter::DrawSolidCircle(rgb_image, pixel_uv.x(), pixel_uv.y(), 3, pixel_color);
                ImagePainter::DrawString(rgb_image, std::to_string(output.features_id[i]), pixel_uv.x(), pixel_uv.y(), pixel_color);
            }

            // Draw image to show.
            Visualizor2D::ShowImage(std::string("frame ") + camera_name[0], rgb_image);
        }

        Visualizor2D::WaitKey(0);
    }
}

void TestFrontendMono(const std::vector<std::string> &cam0_filenames) {
    ReportInfo(">> Test frontend mono.");

    // Config frontend.
    GrayImage image;
    Visualizor2D::LoadImage(cam0_filenames.front(), image);
    visual_frontend::FrontendMono frontend(image.rows(), image.cols());
    frontend.options().kEnableRecordBinaryCurveLog = kEnableRecordCurveLog;
    frontend.options().kEnableRecordBinaryImageLog = kEnableRecordImageLog;
    frontend.options().kEnableShowVisualizeResult = kEnableDrawingTrackingResult;
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 40;
    frontend.Initialize("../../Workspace/output/frontend_mono_log.binlog");

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
    frontend.camera_models().emplace_back(std::make_unique<sensor_model::CameraPinholeRadtan>());
    frontend.camera_models().back()->SetIntrinsicParameter(fx, fy, cx, cy);
    frontend.camera_models().back()->SetDistortionParameter(std::vector<float> {k1, k2, k3, p1, p2});

    // Config image processor.
    // frontend.image_processor() = std::make_unique<image_processor::CensusProcessor>();

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 30;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 10;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 10;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<KltType>();
    frontend.feature_tracker()->options().kMethod = feature_tracker::OpticalFlowMethod::kFast;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 6;
    frontend.feature_tracker()->options().kPatchColHalfSize = 6;
    frontend.feature_tracker()->options().kMaxIteration = 15;

    // Config epipolar solver. Disable it for better performance.
    // frontend.epipolar_solver() = std::make_unique<vision_geometry::EpipolarSolver>();
    // frontend.epipolar_solver()->options().kMethod = vision_geometry::EpipolarSolver::EpipolarMethod::kRansac;
    // frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    float time_stamp_s = 0.0f;
    for (const auto &filename: cam0_filenames) {
        GrayImage image;
        Visualizor2D::LoadImage(filename, image);
        frontend.RunOnce(image, time_stamp_s);
        ShowFrontendMonoOutput(frontend, image);

        time_stamp_s += 0.05f;
    }
}

void ShowFrontendStereoOutput(const visual_frontend::FrontendStereo &frontend, const GrayImage &image_left, const GrayImage &image_right) {
    // Show output data.
    if (kEnableDrawingOutputResult) {
        const auto &output = frontend.output_data();

        // Camera name of each camera is camera_name[camera_id].
        std::vector<std::string> camera_name = {"left", "right"};

        if (image_left.data() != nullptr) {
            // Convert gray image to rgb image.
            RgbImage rgb_image;
            uint8_t *rgb_buf = (uint8_t *)SlamMemory::Malloc(image_left.rows() * image_left.cols() * 3 * sizeof(uint8_t));
            rgb_image.SetImage(rgb_buf, image_left.rows(), image_left.cols(), true);
            ImagePainter::ConvertUint8ToRgb(image_left.data(), rgb_image.data(), image_left.rows() * image_left.cols());

            // Draw all observed features in this frame and this camera image.
            for (uint32_t i = 0; i < output.features_id.size(); ++i) {
                const Vec2 pixel_uv = output.observes_per_frame[i][0].raw_pixel_uv;
                const RgbPixel pixel_color = RgbColor::kCyan;
                ImagePainter::DrawSolidCircle(rgb_image, pixel_uv.x(), pixel_uv.y(), 3, pixel_color);
                ImagePainter::DrawString(rgb_image, std::to_string(output.features_id[i]), pixel_uv.x(), pixel_uv.y(), pixel_color);
            }

            // Draw image to show.
            Visualizor2D::ShowImage(std::string("frame ") + camera_name[0], rgb_image);
        }

        if (image_right.data() != nullptr) {
            // Convert gray image to rgb image.
            RgbImage rgb_image;
            uint8_t *rgb_buf = (uint8_t *)SlamMemory::Malloc(image_right.rows() * image_right.cols() * 3 * sizeof(uint8_t));
            rgb_image.SetImage(rgb_buf, image_right.rows(), image_right.cols(), true);
            ImagePainter::ConvertUint8ToRgb(image_right.data(), rgb_image.data(), image_right.rows() * image_right.cols());

            // Draw all observed features in this frame and this camera image.
            for (uint32_t i = 0; i < output.features_id.size(); ++i) {
                CONTINUE_IF(output.observes_per_frame[i].size() < 2);
                const Vec2 pixel_uv = output.observes_per_frame[i][1].raw_pixel_uv;
                const RgbPixel pixel_color = RgbColor::kCyan;
                ImagePainter::DrawSolidCircle(rgb_image, pixel_uv.x(), pixel_uv.y(), 3, pixel_color);
                ImagePainter::DrawString(rgb_image, std::to_string(output.features_id[i]), pixel_uv.x(), pixel_uv.y(), pixel_color);
            }

            // Draw image to show.
            Visualizor2D::ShowImage(std::string("frame ") + camera_name[1], rgb_image);
        }

        Visualizor2D::WaitKey(0);
    }
}

void TestFrontendStereo(const std::vector<std::string> &cam0_filenames, const std::vector<std::string> &cam1_filenames) {
    ReportInfo(">> Test frontend stereo.");

    // Config frontend.
    GrayImage image;
    Visualizor2D::LoadImage(cam0_filenames.front(), image);
    visual_frontend::FrontendStereo frontend(image.rows(), image.cols());
    frontend.options().kEnableRecordBinaryCurveLog = kEnableRecordCurveLog;
    frontend.options().kEnableRecordBinaryImageLog = kEnableRecordImageLog;
    frontend.options().kEnableShowVisualizeResult = kEnableDrawingTrackingResult;
    frontend.options().kSelfSelectKeyframe = true;
    frontend.options().kMaxStoredFeaturePointsNumber = 100;
    frontend.options().kMinDetectedFeaturePointsNumberInCurrentImage = 30;
    frontend.Initialize("../../Workspace/output/frontend_stereo_log.binlog");

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
    // Left camera model.
    frontend.camera_models().emplace_back(std::make_unique<sensor_model::CameraPinholeRadtan>());
    frontend.camera_models().back()->SetIntrinsicParameter(fx, fy, cx, cy);
    frontend.camera_models().back()->SetDistortionParameter(std::vector<float> {k1, k2, k3, p1, p2});
    // Right camera model.
    frontend.camera_models().emplace_back(std::make_unique<sensor_model::CameraPinholeRadtan>());
    frontend.camera_models().back()->SetIntrinsicParameter(fx, fy, cx, cy);
    frontend.camera_models().back()->SetDistortionParameter(std::vector<float> {k1, k2, k3, p1, p2});

    // Config feature detector.
    frontend.feature_detector() = std::make_unique<FeatureType>();
    frontend.feature_detector()->options().kMinFeatureDistance = 30;
    frontend.feature_detector()->options().kGridFilterRowDivideNumber = 10;
    frontend.feature_detector()->options().kGridFilterColDivideNumber = 10;

    // Config optical flow tracker.
    frontend.feature_tracker() = std::make_unique<KltType>();
    frontend.feature_tracker()->options().kMethod = feature_tracker::OpticalFlowMethod::kFast;
    frontend.feature_tracker()->options().kPatchRowHalfSize = 6;
    frontend.feature_tracker()->options().kPatchColHalfSize = 6;
    frontend.feature_tracker()->options().kMaxIteration = 15;

    // Config epipolar solver.
    // frontend.epipolar_solver() = std::make_unique<vision_geometry::EpipolarSolver>();
    // frontend.epipolar_solver()->options().kMethod = vision_geometry::EpipolarSolver::EpipolarMethod::kRansac;
    // frontend.epipolar_solver()->options().kMaxEpipolarResidual = 3e-2f;

    float time_stamp_s = 0.0f;
    for (unsigned i = 0; i < cam0_filenames.size(); ++i) {
        GrayImage image_left;
        GrayImage image_right;
        Visualizor2D::LoadImage(cam0_filenames[i], image_left);
        Visualizor2D::LoadImage(cam1_filenames[i], image_right);
        frontend.RunOnce(image_left, image_right, time_stamp_s);
        ShowFrontendStereoOutput(frontend, image_left, image_right);

        time_stamp_s += 0.05f;
    }
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test visual frontend on euroc dataset." RESET_COLOR);

    std::vector<std::string> cam0_filenames;

    if (!SlamOperation::GetFilesNameInDirectory("/media/horizon/Database/robotic_datasets/slam/Euroc/MH_01_easy/camera0", cam0_filenames)) {
        if (!SlamOperation::GetFilesNameInDirectory("/mnt/d/My_Github/Datasets/Euroc/MH_01_easy/mav0/cam0/data", cam0_filenames)) {
            RETURN_FALSE_IF_FALSE(SlamOperation::GetFilesNameInDirectory("D:/My_Github/Datasets/Euroc/MH_01_easy/mav0/cam0/data", cam0_filenames));
        }
    }
    std::sort(cam0_filenames.begin(), cam0_filenames.end());

    std::vector<std::string> cam1_filenames;
    if (!SlamOperation::GetFilesNameInDirectory("/media/horizon/Database/robotic_datasets/slam/Euroc/MH_01_easy/camera1", cam1_filenames)) {
        if (!SlamOperation::GetFilesNameInDirectory("/mnt/d/My_Github/Datasets/Euroc/MH_01_easy/mav0/cam1/data", cam1_filenames)) {
            RETURN_FALSE_IF_FALSE(SlamOperation::GetFilesNameInDirectory("D:/My_Github/Datasets/Euroc/MH_01_easy/mav0/cam1/data", cam1_filenames));
        }
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
