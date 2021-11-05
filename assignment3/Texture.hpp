//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen3/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data_rgb;
    cv::Mat image_data_gray;

public:
    Texture(const std::string& name)
    {
        image_data_rgb = cv::imread(name);
        cv::cvtColor(image_data_rgb, image_data_rgb, cv::COLOR_RGB2BGR);
        cv::cvtColor(image_data_rgb, image_data_gray, cv::COLOR_BGR2GRAY);
        width = image_data_rgb.cols;
        height = image_data_rgb.rows;
    }

    int width, height;

    Eigen::Vector3f getColorRGB(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data_rgb.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    float getColorGray(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data_gray.at<uchar>(v_img, u_img);
        return color;
    }

};
#endif //RASTERIZER_TEXTURE_H
