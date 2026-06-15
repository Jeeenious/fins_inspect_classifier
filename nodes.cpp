/*******************************************************************************
 * nodes.cpp — fins_inspect_classifier 插件入口
 ******************************************************************************/

#include "yolo_nodes.hpp"
#include <fins/type/type_register.hpp>
#include <opencv2/core/mat.hpp>

DEFINE_PLUGIN_ENTRY(fins::STATELESS)

REGISTER_PLUGIN_INIT({
  FINS_TYPE_REGISTER.register_type<cv::Mat>("cv::Mat");
})
