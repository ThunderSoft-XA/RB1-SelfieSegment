#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "inference_tf.hpp"

using namespace inference;

int main(int argcm, char **argv)
{
    Settings seg_setting;

    seg_setting.model_name = "./selfie_segmentation_landscape.tflite";
    seg_setting.input_bmp_name = "./selfie_test.png";

    cv::Mat source_mat;
    source_mat = cv::imread(seg_setting.input_bmp_name);

    tf::TFInference seg_inference;

    seg_inference.setSettings(&seg_setting);

    seg_inference.loadModel();
    seg_inference.setInferenceParam();

    cv::Mat input_mat = cv::imread(seg_setting.input_bmp_name);

    tf::InputPair seg_input_pair(0,input_mat);
    tf::InputPairVec seg_input_pair_vec;//(seg_inference.getInputsNum());
    seg_input_pair_vec.push_back(seg_input_pair);

    seg_inference.loadData(seg_input_pair_vec);

    std::vector<float> seg_result_vec;
    seg_inference.inferenceModel<float>(seg_result_vec);

    std::cout << "seg_result_vec size : " << seg_result_vec.size() << std::endl;

    cv::Mat result_mat(cv::Size(256,144),CV_8UC1);
    int piexl_index = 0;
    for(auto piexl_value : seg_result_vec) {
        result_mat.data[piexl_index] = piexl_value * 255;
        piexl_index++;
    }
    cv::resize(source_mat,source_mat,cv::Size(256,144));
    cv::Mat result_mat_rgb;
    source_mat.copyTo(result_mat_rgb, result_mat);
    cv::imwrite("./result_mat.png",result_mat);
    cv::imwrite("./result_mat_rgb.png",result_mat_rgb);

    // std::vector<std::vector<cv::Point>> contours;
    // std::vector<cv::Vec4i> hierarchy;
    // //此处输入的图像必须是一个二值的单通道图像（src），否则findContours不执行
    // cv::findContours(result_mat, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    // int index = 0;
    // for (; index >= 0; index = hierarchy[index][0]) {
    //     //随机生成不同的颜色值
    //     cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
    //     //将轮廓绘制在预先建立好的mat中
    //     cv::drawContours(result_mat_rgb, contours, index, color, cv::FILLED, 8, hierarchy);
    // }

    cv::Mat background = cv::imread("./background.jpg");
    cv::Mat imageROI = background(cv::Rect(background.cols / 2 - result_mat_rgb.cols /2 ,
            background.rows - result_mat_rgb.rows,result_mat_rgb.cols,result_mat_rgb.rows));   //获取感兴趣区域，即logo要放置的区域
    cv::addWeighted(imageROI,0.9,result_mat_rgb,0.6,0,imageROI);     //图像叠加

    cv::imwrite("./final.png",background);

    return 0;

}