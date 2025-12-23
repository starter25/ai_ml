# src/train_baseline.py
import os
import sys
import platform
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ===== ROOT 세팅 =====
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import load_dataset
from build_model import build_mlp

# ===== 저장 경로(기존 baseline 모델 폴더) =====
MODEL_PATH = os.path.join(ROOT, "experiments", "height_mlp_model")


def open_file(path: str):
    """Windows면 파일 자동 열기"""
    try:
        if platform.system().lower().startswith("win"):
            os.startfile(path)  # type: ignore
    except Exception as e:
        print(f"[WARN] Could not open file: {path} ({e})")


def save_history_plots(history, out_dir: str, auto_open: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    hist_df = pd.DataFrame(history.history)

    # 1) CSV 저장
    csv_path = os.path.join(out_dir, "history.csv")
    hist_df.to_csv(csv_path, index=False)
    print(f"Saved history CSV -> {csv_path}")

    # 2) loss 그래프
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.figure()
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df.columns:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Baseline Loss Curve (MSE)")
    plt.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {loss_png}")

    # 3) mae 그래프 (baseline은 metrics=["mae"]라서 보통 mae/val_mae가 존재)
    mae_png = os.path.join(out_dir, "mae_curve.png")
    if "mae" in hist_df.columns:
        plt.figure()
        plt.plot(hist_df["mae"], label="train_mae")
        if "val_mae" in hist_df.columns:
            plt.plot(hist_df["val_mae"], label="val_mae")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("mae")
        plt.title("Baseline MAE Curve")
        plt.savefig(mae_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {mae_png}")
    else:
        mae_png = None
        print("[WARN] 'mae' not found in history. (Check compile metrics)")

    # 4) 자동 열기
    if auto_open:
        open_file(loss_png)
        if mae_png is not None:
            open_file(mae_png)


def save_model_structure(model: tf.keras.Model, out_dir: str, auto_open: bool = True):
    """모델 구조 그림 저장(가능하면)"""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "model_structure.png")
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=png_path,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=200,
        )
        print(f"Saved model structure -> {png_path}")
        if auto_open:
            open_file(png_path)
    except Exception as e:
        print("[WARN] plot_model failed (pydot/graphviz 미설치일 가능성 큼).")
        print(" - 해결: pip install pydot")
        print(" - Graphviz 설치 후 PATH 설정")
        print(f" - error: {e}")


def train():
    # 1) 데이터 로딩
    X, Y = load_dataset()
    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")

    # 2) 모델
    model = build_mlp(input_dim=X.shape[1], output_dim=Y.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    # 3) 학습 (baseline 그대로)
    history = model.fit(
        X, Y,
        batch_size=128,
        epochs=50,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )

    # 4) 저장 (baseline 그대로)
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    # 5) 시각화 저장
    save_history_plots(history, out_dir=MODEL_PATH, auto_open=True)
    save_model_structure(model, out_dir=MODEL_PATH, auto_open=True)

    print("Done.")


if __name__ == "__main__":
    train()
