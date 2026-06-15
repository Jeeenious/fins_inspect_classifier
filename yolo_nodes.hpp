/*******************************************************************************
 * yolo_nodes.hpp — MeterClassifier (cv::Mat frame → 固定槽位输出)
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <fins/node.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "config.h"

struct Detection {
  cv::Rect bbox;
  int class_id = -1;
  float confidence = 0;
  std::string class_name;
};

class MeterClassifier : public fins::Node {
public:
  void define() override {
    set_name("MeterClassifier");
    set_description("仪表分类: YOLO ONNX → 固定槽位 light_0..3, twist_0..3, gauge_0..3, digital_0..3");
    set_category("Inspect");
    register_input<cv::Mat>("frame", &MeterClassifier::on_frame);
    for (int i = 0; i < config::LIGHT_SLOTS;   ++i) register_output<cv::Mat>("light_"   + std::to_string(i));
    for (int i = 0; i < config::TWIST_SLOTS;   ++i) register_output<cv::Mat>("twist_"   + std::to_string(i));
    for (int i = 0; i < config::GAUGE_SLOTS;   ++i) register_output<cv::Mat>("gauge_"   + std::to_string(i));
    for (int i = 0; i < config::DIGITAL_SLOTS; ++i) register_output<cv::Mat>("digital_" + std::to_string(i));
    register_output<cv::Mat>("preview");
  }
  void initialize() override {
    logger->info("MeterClassifier 加载模型: {}", config::MODEL_PATH);
    net_ = cv::dnn::readNetFromONNX(config::MODEL_PATH);
    if (net_.empty()) logger->warn("ONNX 模型加载失败");
    else logger->info("MeterClassifier 模型加载成功");
  }
  void run() override {}

  void on_frame(const cv::Mat &frame, fins::AcqTime) {
    if (net_.empty() || frame.empty()) return;

    auto dets = infer(frame);
    // 按类别分组, 每组从左到右从上到下排序
    std::map<std::string, std::vector<Detection>> groups;
    for (auto &d : dets) groups[d.class_name].push_back(d);

    int total = 0;
    for (auto &[label, vec] : groups) {
      auto [prefix, slots] = config::SLOT_INFO(label);
      if (slots <= 0) continue;
      std::sort(vec.begin(), vec.end(), [](const Detection &a, const Detection &b) {
        int ay = a.bbox.y + a.bbox.height/2, by = b.bbox.y + b.bbox.height/2;
        return ay != by ? ay < by : a.bbox.x < b.bbox.x;
      });
      for (size_t i = 0; i < vec.size() && (int)i < slots; ++i) {
        auto &d = vec[i];
        cv::Rect roi(d.bbox);
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.area() <= 0) continue;
        logger->info("检出: {} #[{}] [{},{},{},{}] conf={:.2f}", label, i,
                     roi.x, roi.y, roi.width, roi.height, d.confidence);
        send(prefix + "_" + std::to_string(i), frame(roi).clone());
        ++total;
      }
    }
    // 预览图每秒发一次（减少负载）
    if (++tick_ % 10 == 0) {
      cv::Mat preview = frame.clone();
      for (auto &d : dets) {
        cv::Scalar color;
        if (d.class_name == "led")        color = cv::Scalar(0, 255, 0);    // 绿
        else if (d.class_name == "twist")   color = cv::Scalar(0, 255, 255);  // 黄
        else if (d.class_name == "gauge")   color = cv::Scalar(255, 0, 0);    // 蓝
        else                                color = cv::Scalar(0, 0, 255);    // 红
        cv::rectangle(preview, d.bbox, color, 2);
      }
      send("preview", preview);
    }
    if (tick_ % 30 == 0)
      logger->info("分类器: 帧内检出 {} 个目标", total);
  }

private:
  cv::dnn::Net net_;
  int tick_ = 0;

  std::vector<Detection> infer(const cv::Mat &frame) {
    // preprocess
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0,
      cv::Size(config::INPUT_SIZE, config::INPUT_SIZE), cv::Scalar(), true, false);
    net_.setInput(blob);
    auto outputs = net_.forward();

    // YOLOv8 output: (1, 4+nc, 8400) — reshape+transpose 成 (8400, 4+nc)
    // outputs.size[1]=4+nc, size[2]=8400, reshape(1,channels) → (7, 8400) → t()
    cv::Mat out = outputs.reshape(1, outputs.size[1]).t();
    int num_dets    = out.rows;  // 8400
    int num_outputs = out.cols;
    int num_classes = num_outputs - 4;

    float x_scale = (float)frame.cols / config::INPUT_SIZE;
    float y_scale = (float)frame.rows / config::INPUT_SIZE;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    for (int i = 0; i < num_dets; ++i) {
      float *row = out.ptr<float>(i);
      float cx = row[0] * x_scale;
      float cy = row[1] * y_scale;
      float w  = row[2] * x_scale;
      float h  = row[3] * y_scale;

      float max_conf = 0;
      int best_cls = -1;
      for (int c = 0; c < num_classes; ++c) {
        float score = 1.0f / (1.0f + std::exp(-row[4 + c]));  // sigmoid
        if (score > max_conf) { max_conf = score; best_cls = c; }
      }
      if (max_conf < config::CONF_THRESHOLD) continue;

      boxes.emplace_back(static_cast<int>(cx - w/2), static_cast<int>(cy - h/2),
                         static_cast<int>(w), static_cast<int>(h));
      confs.push_back(max_conf);
      class_ids.push_back(best_cls);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, config::CONF_THRESHOLD, config::NMS_THRESHOLD, indices);

    std::vector<Detection> result;
    for (int idx : indices) {
      Detection d;
      d.bbox = boxes[idx];
      d.confidence = confs[idx];
      d.class_id = class_ids[idx];
      int n = sizeof(config::CLASS_NAMES) / sizeof(config::CLASS_NAMES[0]);
      d.class_name = (d.class_id >= 0 && d.class_id < n)
                     ? config::CLASS_NAMES[d.class_id] : "unknown";
      result.push_back(d);
    }
    std::sort(result.begin(), result.end(),
              [](const Detection &a, const Detection &b) { return a.confidence > b.confidence; });
    return result;
  }
};
EXPORT_NODE(MeterClassifier)
