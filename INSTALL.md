# Installation Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU
- `nvcc` compiler（版本需與 PyTorch 的 CUDA 版本一致）

確認方式：
```bash
nvcc --version
python3 -c "import torch; print(torch.version.cuda)"
```

---

## Step 1：安裝 Python 依賴

```bash
pip install torch torchvision  # 需與系統 CUDA 版本對應，參考 https://pytorch.org
pip install -r requirements.txt
```

`requirements.txt` 包含：`tensorboard`, `numpy`, `scipy`, `open3d>=0.8`, `Pillow`, `tqdm`

---

## Step 2：編譯 CUDA Extensions

### PointNet2
```bash
cd pointnet2
python setup.py install
cd ..
```

### KNN
```bash
cd knn
python setup.py install
cd ..
```

---

## Step 3：安裝 graspnetAPI

graspnetAPI 提供 `GraspGroup`（推論核心資料結構）與 `GraspNetEval`（benchmark 評估）。

### 標準安裝（從 PyPI）
```bash
pip install graspnetAPI
```

### 從原始碼安裝
```bash
cd ../graspnetAPI
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install .
```

### 驗證
```bash
python3 -c "from graspnetAPI import GraspGroup; print('OK')"
```

---

## Step 4：驗證整體安裝

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path checkpoint-rs.tar
```

---

## 常見錯誤與解法

### 1. `error: Cannot update time stamp of directory 'graspnetAPI.egg-info'`

**原因：** 舊的 `.egg-info` 目錄由 root 建立，當前使用者無寫入權限（常見於 Docker/Dev Container）。

**解法：**
```bash
sudo rm -rf graspnetAPI.egg-info
pip install .
```

---

### 2. `The 'sklearn' PyPI package is deprecated`

**原因：** graspnetAPI 依賴已棄用的 `sklearn` 套件名稱，新版 pip 預設封鎖安裝。

**解法：**
```bash
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install .
```

---

### 3. `error: could not delete 'build/lib/graspnetAPI/__init__.py': Permission denied`

**原因：** `build/` 目錄由 root 建立，當前使用者無刪除權限。

**解法：**
```bash
sudo rm -rf build/
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install .
```

---

### 4. `numpy-quaternion requires numpy>=1.25, but you have numpy 1.23.4`

**原因：** graspnetAPI 強制安裝 `numpy==1.23.4`，與容器中其他套件版本衝突。

**影響：** graspnet 推論（`GraspGroup`、ROS2 node）本身不受影響。但若其他套件依賴 `numpy-quaternion`，可能出錯。

**確認 graspnet 是否正常：**
```bash
python3 -c "from graspnetAPI import GraspGroup; print('OK')"
```

---

## Checkpoints

| 檔案 | 相機 | 備註 |
|---|---|---|
| `checkpoint-rs.tar` | RealSense | 遷移新場景效果較好，建議優先使用 |
| `checkpoint-kn.tar` | Kinect | — |
