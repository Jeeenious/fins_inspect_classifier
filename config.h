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
// 后续加标: 在末尾追加, 重新训练+导出即可
inline const std::string CLASS_NAMES[] = {
  "led", "gauge", "digital"
};

// label → 输出端口
inline const std::string PORT_MAP(const std::string &label) {
  if (label == "led")     return "light";
  if (label == "gauge")   return "gauge";
  if (label == "digital") return "digital";
  return "";
}

}  // namespace config
