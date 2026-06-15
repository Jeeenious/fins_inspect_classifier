# fins_classifier — 仪表分类

MeterClassifier 节点：YOLOv8 ONNX → 固定槽位输出裁剪图。

## 输出端口

| 类别 | 槽位数 | 端口名 | 排序 |
|------|--------|--------|------|
| LED 指示灯 | 8 | `light_0` .. `light_7` | 从上到下、从左到右 |
| 指针表 | 4 | `gauge_0` .. `gauge_3` | 从上到下、从左到右 |
| 数显 | 4 | `digital_0` .. `digital_3` | 从上到下、从左到右 |

未检出的槽位悬空不发送。

## 管线

```
CameraSource → MeterClassifier → light_0  → LEDAnalyzer_0 → color + freq + active
                               → light_1  → LEDAnalyzer_1
                               → ...
                               → gauge_0  → GaugeReader_0
                               → digital_0 → DigitalReader_0
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
    └── valid/images/      2246 张
```

### 新增自定义类别

1. 在 `dataset/` 下创建新数据集目录，按 YOLO 格式放 images + labels
2. 合并到 `dataset/merged/`，class id 按 `CLASS_NAMES` 顺序递推
3. 更新 `data.yaml` 的 `names` 和 `nc`
4. 更新 `config.h` 的 `CLASS_NAMES`、`SLOT_INFO` 和输出端口

## 训练

```bash
cd inspect/fins_classifier
pip install ultralytics "numpy<2"

yolo detect train \
  data=dataset/merged/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=320 \
  device=cpu
```

## 验证

```bash
yolo detect val model=runs/detect/train-2/weights/best.pt data=dataset/merged/data.yaml
yolo detect predict model=runs/detect/train-2/weights/best.pt source=dataset/digital_display/valid/images/ save=True
```

## 导出 & 部署

```bash
yolo detect export model=runs/detect/train-2/weights/best.pt format=onnx imgsz=320
cp runs/detect/train-2/weights/best.onnx model/best.onnx
fins build inspect/fins_classifier
```

`CMakeLists.txt` 编译时注入 `MODEL_DIR` 绝对路径，运行时自动定位 `model/best.onnx`。
