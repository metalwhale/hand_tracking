## Hand tracking

### 1. ファイル説明
- `palm_detection_without_custom_op.tflite`（手のひら検出）モデルファイル：[*mediapipe-models*]レポジトリよりダウンロードしました。
- `hand_landmark.tflite`（ランドマーク検出）モデルファイル：[*mediapipe*]レポジトリよりダウンロードしました。
- `anchors.csv`ファイルと`hand_tracker.py`ファイル：[*hand_tracking*]レポジトリよりダウンロードしました。

### 2. 実施方法
```
$ pip install opencv-python tensorflow
$ python run.py
```

### 3. 結果
![Result](/output.gif?raw=true "Result")

[*mediapipe-models*]: https://github.com/junhwanjang/mediapipe-models/tree/master/palm_detection/mediapipe_models
[*mediapipe*]: https://github.com/google/mediapipe/tree/master/mediapipe/models
[*hand_tracking*]: https://github.com/wolterlw/hand_tracking
