#include "frontend.h"
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

    cv::Mat image = cv::imread(cam0_filenames.front());
    VisualFrontend::Frontend frontend(image.rows, image.cols);
    LogDebug("Frontend config image size is " << image.rows << ", " << image.cols);

    for (const auto &filename : cam0_filenames) {
        cv::Mat cv_image = cv::imread(filename, 0);
        Image image;
        image.SetImage(cv_image.data, cv_image.rows, cv_image.cols);
        frontend.RunOnce(image);
    }


    return 0;
}
