import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse


def short_name(model_str: str) -> str:
    p = Path(model_str)
    s = model_str
    if "weights" in model_str:
        # runs/detect/<name>/weights/best.pt -> <name>
        try:
            s = p.parent.parent.name
        except Exception:
            s = p.stem
    elif p.suffix in (".pt", ".pth"):
        # if just a file like yolo26n.pt -> stem
        # if path like runs/detect/.../model.pt -> parent name
        if p.parent.name and p.parent.name != ".":
            s = p.parent.name
        else:
            s = p.stem
    else:
        s = p.name or model_str
    # remove extensions
    s = s.replace(".pt", "").replace(".pth", "")
    return s


def load_metrics(json_path: Path):
    data = json.loads(json_path.read_text())
    entries = []
    for item in data:
        model_raw = item.get("model") or item.get("model_path") or "unknown"
        name = short_name(model_raw)
        metrics = item.get("metrics", {})
        entries.append({
            "model_raw": model_raw,
            "name": name,
            "F1_Score": metrics.get("F1_Score", metrics.get("F1", None)),
            "mAP@50": metrics.get("mAP@50", metrics.get("mAP50", None)),
            "mAP@50-95": metrics.get("mAP@50-95", metrics.get("mAP50_95", None)),
        })
    return entries


def plot_metrics(entries, out_path: Path = None):
    # entries: list of dicts with keys name,F1_Score,mAP@50,mAP@50-95
    entries = sorted(entries, key=lambda x: x.get("F1_Score", 0), reverse=True)
    names = [e["name"] for e in entries]
    f1 = [e.get("F1_Score") for e in entries]
    m50 = [e.get("mAP@50") for e in entries]
    m5095 = [e.get("mAP@50-95") for e in entries]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.0), 6))
    ax.bar(x - width, f1, width, label="F1 Score")
    ax.bar(x, m50, width, label="mAP@50")
    ax.bar(x + width, m5095, width, label="mAP@50-95")

    ax.set_ylabel("Score", fontsize = 12)
    ax.set_title("Model metrics", fontsize = 14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize = 12)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved metrics plot to {out_path}")
    else:
        fig.show()


def plot_map50_comparison(entries_a, entries_b, label_a="clean", label_b="corrupted", out_path: Path = None):
    # build dict by short name
    dict_a = {e["name"]: e for e in entries_a}
    dict_b = {e["name"]: e for e in entries_b}
    common = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if not common:
        print("No common models found between the two JSONs to compare.")
        return

    a_vals = [dict_a[n].get("mAP@50", 0) for n in common]
    b_vals = [dict_b[n].get("mAP@50", 0) for n in common]

    x = np.arange(len(common))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(common) * 1.0), 6))
    ax.bar(x - width / 2, a_vals, width, label=label_a)
    ax.bar(x + width / 2, b_vals, width, label='augmented')

    ax.set_ylabel("mAP@50", fontsize = 12)
    ax.set_title(f"mAP@50: {label_a} vs augmented test images", fontsize = 14)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha="right", fontsize = 12)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved comparison plot to {out_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=r"model_results/test_results/test_metrics_summary.json", help="metrics JSON to plot")
    parser.add_argument("--corrupted", type=str, default=r"model_results/corrupted_test_results/test_metrics_summary.json", help="corrupted metrics JSON to compare")
    parser.add_argument("--out-dir", type=str, default=".", help="output directory for saved plots")
    args = parser.parse_args()

    repo = Path(__file__).parent
    file_a = repo / args.file
    file_b = repo / args.corrupted
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_metrics(file_a)
    plot_metrics(entries, out_path=out_dir / "metrics_bar.png")

    entries_b = load_metrics(file_b)
    plot_map50_comparison(entries, entries_b, label_a="clean", label_b="corrupted", out_path=out_dir / "map50_comparison.png")


if __name__ == "__main__":
    main()
