# fins_classifier — 仪表分类

MeterClassifier 节点：YOLOv8 ONNX → 固定槽位输出裁剪图。

## 输出端口

| 类别 | 槽位数 | 端口名 | 排序 |
|------|--------|--------|------|
| 按钮灯 | 4 | `light_0` .. `light_3` | 从上到下、从左到右 |
| 旋钮 | 4 | `twist_0` .. `twist_3` | 从上到下、从左到右 |
| 指针表 | 4 | `gauge_0` .. `gauge_3` | 从上到下、从左到右 |
| 数显 | 4 | `digital_0` .. `digital_3` | 从上到下、从左到右 |

未检出的槽位悬空不发送。

## 管线

```
CameraSource → MeterClassifier → light_0   → LEDAnalyzer_0  → color + freq + active
                               → light_1   → LEDAnalyzer_1
                               → ...
                               → twist_0   → TwistReader_0
                               → gauge_0   → GaugeReader_0
                               → digital_0 → DigitalReader_0
```

## 数据集

已准备四份源数据集，合并后可直接训练：

```
dataset/
├── light_button/      train 192 + valid 40, 类别 "led" + "twist"
├── analog_gauge/      train 6396 + valid 1829, 类别 "gauge"(仅采样 800 张参与合并)
├── digital_display/   train 537 + valid 121, 类别 "digital"
└── merged/            train 5120 + valid 617, 4 类合并
    ├── data.yaml      names: [led, twist, gauge, digital]
    ├── train/images/  5120 张 (led 3711, gauge 640, digital 577, twist 192)
    └── valid/images/  617 张 (led 296, gauge 160, digital 121, twist 40)
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
