#include "frontend.h"
#include "slam_memory.h"
#include "slam_operations.h"

namespace VISUAL_FRONTEND {

Frontend::Frontend(const uint32_t image_rows, const uint32_t image_cols) {
    options_.kImageRows = image_rows;
    options_.kImageCols = image_cols;
}

bool Frontend::Initialize(const std::string &log_file_name) {
    // Allocate memory for stored buffer.
    const uint32_t size = options_.kImageRows * options_.kImageCols;
    stored_buff_ = (uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * size * 8);
    RETURN_FALSE_IF(stored_buff_ == nullptr);
    uint8_t *buf = stored_buff_;
    for (uint32_t i = 0; i < 4; ++i) {
        stored_pyramids_[i].SetRawImage(buf, options_.kImageRows, options_.kImageCols);
        buf += size;
        stored_pyramids_[i].SetPyramidBuff(buf);
        buf += size;
    }

    // Reserve capacity for output data.
    output_data_.features_id.reserve(options_.kMaxStoredFeaturePointsNumber);
    output_data_.observes_per_frame.reserve(options_.kMaxStoredFeaturePointsNumber);

    // Register packages for log file.
    if (options_.kEnableRecordBinaryCurveLog) {
        if (!logger_.CreateLogFile(log_file_name)) {
            ReportError("Visual frontend cannot create log file.");
            options_.kEnableRecordBinaryCurveLog = false;
            return false;
        }

        // This should be implemented by sub class.
        RegisterLogPackages();

        // Write log file header.
        logger_.PrepareForRecording();
    }

    return true;
}

Frontend::~Frontend() {
    if (stored_buff_ == nullptr) {
        SlamMemory::Free(stored_buff_);
    }
}

bool Frontend::CheckAllComponents() {
    RETURN_FALSE_IF(feature_tracker_ == nullptr);
    RETURN_FALSE_IF(feature_detector_ == nullptr);
    RETURN_FALSE_IF(camera_model_ == nullptr);
    return true;
}

}
