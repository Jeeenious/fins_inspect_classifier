/*******************************************************************************
 * config.h — YOLO ONNX 参数
 ******************************************************************************/
#pragma once

#include <string>

namespace config {

// MODEL_DIR 由 CMake 编译时注入绝对路径
inline const std::string MODEL_PATH = std::string(MODEL_DIR) + "/best.onnx";
inline constexpr int    INPUT_SIZE = 320;
inline constexpr float  CONF_THRESHOLD = 0.7f;
inline constexpr float  NMS_THRESHOLD  = 0.4f;

// YOLO 训练类别 (须与 ONNX 输出通道严格一致)
inline const std::string CLASS_NAMES[] = {"led", "twist", "gauge", "digital"};

// label → 输出端口名
inline std::string VEC_PORT(const std::string &label) {
  if (label == "led")     return "leds";
  if (label == "twist")   return "twists";
  if (label == "gauge")   return "gauges";
  if (label == "digital") return "digitals";
  return "";
}

}  // namespace config
