#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "inference_tf.hpp"
#include "camera2appsink.hpp"
#include "appsrc2rtsp.hpp"

using namespace inference;

extern Queue<cv::Mat> rgb_mat_queue;

int main(int argcm, char **argv)
{
    std::string json_file = "./gst_selfie_config.json";

    /* Initialize GStreamer */
    gst_init (nullptr, nullptr);
    GMainLoop *main_loop;  /* GLib's Main Loop */
    /* Create a GLib Main Loop and set it to run */
    main_loop = g_main_loop_new (NULL, FALSE);
    Queue<cv::Mat> mat_queue;
    CameraPipe camera_pipe(json_file);

    camera_pipe.initPipe();
    camera_pipe.rgb_mat_queue = mat_queue;

    camera_pipe.checkElements();

    camera_pipe.setProperty();

    camera_pipe.runPipeline();
    Settings seg_setting;

    seg_setting.model_name = "./selfie_segmentation_landscape.tflite";
    seg_setting.input_bmp_name = "./selfie_test.png";

    tf::TFInference seg_inference;

    seg_inference.setSettings(&seg_setting);

    seg_inference.loadModel();
    seg_inference.setInferenceParam();

    g_print("json_file : %s\n",json_file.c_str());
	Queue<cv::Mat> push_queue;
	APPSrc2RtspSink push_pipe(json_file);
	if( push_pipe.initPipe() == -1 ) {
		return 0;
	}

	push_pipe.push_mat_queue = push_queue;

	if(push_pipe.checkElements() == -1) {
		return 0;
	}

	push_pipe.setProperty();

    std::thread([&](){
        static int count = 0;
        while(1) {
            if(camera_pipe.rgb_mat_queue.empty()) {
                continue;
            }
            cv::Mat input_mat = camera_pipe.rgb_mat_queue.pop();
            std::stringstream input_name;
            input_name << "./input/" << count << ".png";
            cv::imwrite(input_name.str(),input_mat);

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
                g_print("%d\t", result_mat.data[piexl_index]);
                if(piexl_index % 10 == 0) {
                    g_print("\n");
                }
                piexl_index++;
            }
            std::stringstream result_name;
            result_name << "./mask/" << count << ".png";
            cv::imwrite(result_name.str(),result_mat);

            cv::threshold(result_mat, result_mat, 200, 255, cv::THRESH_BINARY);
            cv::resize(input_mat,input_mat,cv::Size(256,144), cv::INTER_NEAREST);
            cv::Mat result_mat_rgb;
            cv::Mat element;
            element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::erode(result_mat, result_mat, element);
            element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::dilate(result_mat, result_mat, element);
            input_mat.copyTo(result_mat_rgb, result_mat);
            cv::imwrite("./result_mat.png",result_mat);
            cv::imwrite("./result_mat_rgb.png",result_mat_rgb);

            cv::Mat background = cv::imread("./background.jpg");
            cv::Mat roi_background;
            cv::Mat imageROI = background(cv::Rect(background.cols / 2 - result_mat_rgb.cols / 2 ,
                    background.rows - result_mat_rgb.rows,result_mat_rgb.cols,result_mat_rgb.rows));   //获取感兴趣区域，即logo要放置的区域
            result_mat_rgb.copyTo(imageROI, result_mat);
            std::stringstream name;
            name << "./img/final_" << count << ".png";
            cv::imwrite(name.str(),background);
            count++;

            cv::resize(background,background,{640,480});

            push_pipe.push_mat_queue.push_back(background);
        }
    }).detach();

    sleep(4);
    std::thread([&](){
		push_pipe.runPipe();
    }).detach();

    g_main_loop_run(main_loop);

    camera_pipe.~CameraPipe();
    g_main_loop_unref (main_loop);

    return 0;

}