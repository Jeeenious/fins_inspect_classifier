/*******************************************************************************
 * config.h — YOLO ONNX 参数
 ******************************************************************************/
#pragma once

#include <string>

namespace config {

// MODEL_DIR 由 CMake 编译时注入绝对路径
inline const std::string MODEL_PATH = std::string(MODEL_DIR) + "/best.onnx";
inline constexpr int    INPUT_SIZE = 320;
inline constexpr float  CONF_THRESHOLD = 0.5f;
inline constexpr float  NMS_THRESHOLD  = 0.4f;

// YOLO 训练类别 (须与 ONNX 输出通道严格一致)
inline const std::string CLASS_NAMES[] = {"led", "gauge", "digital"};

// 每个类别的固定输出槽位数 (检测结果从左到右、从上到下填入, 多余丢弃)
inline constexpr int LIGHT_SLOTS  = 8;
inline constexpr int GAUGE_SLOTS  = 4;
inline constexpr int DIGITAL_SLOTS = 4;

// label → 槽位前缀 + 数量
inline std::pair<std::string, int> SLOT_INFO(const std::string &label) {
  if (label == "led")     return {"light",  LIGHT_SLOTS};
  if (label == "gauge")   return {"gauge",  GAUGE_SLOTS};
  if (label == "digital") return {"digital", DIGITAL_SLOTS};
  return {"", 0};
}

}  // namespace config
