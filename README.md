# fins_inspect_classifier — 目标分类

RegionClassifier 节点：YOLOv8 ONNX → 检测 + 裁剪。

## 端口

| 端口      | 方向  | 类型      | 内容            |
| ------- | --- | ------- | ------------- |
| frame   | in  | cv::Mat | 相机原始帧         |
| light   | out | cv::Mat | led 类别裁剪图     |
| digital | out | cv::Mat | digital 类别裁剪图 |
| gauge   | out | cv::Mat | gauge 类别裁剪图   |

## 管线

```
CameraSource → RegionClassifier → light → LEDAnalyzer → color + freq + active
                                → digital → (后续)
                                → gauge → (后续)
```

## 数据集

已准备三份数据集，合并后可直接训练：

```
dataset/
├── rgb_led_panel/        3564 张, 类别 "led"
├── analog_gauge/         6396 张, 类别 "gauge"
├── digital_display/       698 张, 类别 "digital"
└── merged/              11598 train + 2246 valid, 3 类合并
    ├── data.yaml          names: [led, gauge, digital]
    ├── train/images/      11598 张
    ├── train/labels/      11598 个 txt 标注
    ├── valid/images/      2246 张
    └── valid/labels/      2246 个 txt 标注
```

### 新增自定义类别

1. 在 `dataset/` 下创建新数据集目录，按 YOLO 格式放 images + labels
2. 在 `dataset/merged/` 追加图片和标注（标注 class id 按 CLASS_NAMES 顺序递推）
3. 更新 `data.yaml` 的 `names` 和 `nc`
4. 更新 `config.h` 的 `CLASS_NAMES` 和 `PORT_MAP`

## 训练

```bash
# 在项目根目录 (finevision-plugins) 下运行，或先 cd 到插件目录
cd inspect/fins_inspect_classifier
pip install ultralytics "numpy<2"

yolo detect train \
  data=dataset/merged/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  device=cpu

# 显存不够: batch=8
# 断点续训: resume=True
```

## 验证

```bash
# 看指标 (mAP50 > 0.8 即可用)
yolo detect val model=runs/detect/train-2/weights/best.pt data=dataset/merged/data.yaml

# 实测效果
yolo detect predict model=runs/detect/train-2/weights/best.pt source=dataset/digital_display/valid/images/ save=True
# 结果在 runs/detect/predict/
```

## 导出 & 部署

```bash
# 1. 导出 ONNX (imgsz 必须与训练一致)
yolo detect export \
  model=runs/detect/train-2/weights/best.pt \
  format=onnx \
  imgsz=320

# 2. 模型放 model/ 目录
cp runs/detect/train-2/weights/best.onnx model/best.onnx
```

`CMakeLists.txt` 编译时注入 `MODEL_DIR` 绝对路径到 `config.h`，运行时自动定位 `model/best.onnx`，无需手动改路径。

## 调用

```
fins build inspect/fins_inspect_classifier
```

FINS 管线中连接：

```
CameraSource → RegionClassifier → light  → LEDAnalyzer → color + freq + active
                                → digital → (后续)
                                → gauge   → (后续)
```

模型文件 `model/` 和 `dataset/` 不提交 Git，见 `.gitignore`。
