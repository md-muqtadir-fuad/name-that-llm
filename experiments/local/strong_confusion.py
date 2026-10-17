"""Per-class recall / confusions of the public frozen-LLM plain blend OOF, and with oracle-group Hungarian."""
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from common import OUT, ROOT, CLASSES, K, LAB, load_train, bacc, log_softmax
from structure import hungarian, center

df = load_train(); y = df.y.values; g = df.g.values
zz = np.load(ROOT / "research/ext_oof/q35heads/oof_predictions.npz", allow_pickle=False)
names = list(zz["component_names"])
w = json.loads((ROOT / "research/ext_oof/q35heads/bundle_manifest.json").read_text())["weights"]
wp = {k: v for k, v in w.items() if "center" not in k}
L = log_softmax(sum(v * zz["components"][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()) * 1.72)
lines = ["### Strong plain blend (public frozen Qwen3.5-9B/Qwen3-14B/Qwen3-8B/DeBERTa/RoBERTa/SVM heads, OOF)\n"]
labs = np.array([LAB[c] for c in CLASSES])
for name, pred in [("argmax", L.argmax(1)), ("oracle-prompt Hungarian (full 18)", hungarian(L, g))]:
    cm = confusion_matrix(y, pred, labels=range(K))
    rec = cm.diagonal() / cm.sum(1)
    lines.append(f"\n**{name}**: bacc {bacc(y, pred):.4f}, lab-level bacc {bacc(labs[y], labs[pred]):.4f}, "
                 f"errors within true lab {np.mean(labs[pred[pred!=y]] == labs[y[pred!=y]])*100:.0f}%\n")
    lines.append("| model | recall | top confusions |\n|---|---|---|")
    for i, c in enumerate(CLASSES):
        row = cm[i].astype(float) / cm[i].sum(); row[i] = 0
        top = np.argsort(-row)[:3]
        lines.append(f"| {c} | {rec[i]:.2f} | " + ", ".join(f"{CLASSES[j]} {row[j]*100:.0f}%" for j in top) + " |")
    pairs = sorted(((cm[i, j] + cm[j, i], CLASSES[i], CLASSES[j]) for i in range(K) for j in range(i + 1, K)), reverse=True)
    lines.append("\nTop confused pairs: " + ", ".join(f"{a}<->{b} {n}" for n, a, b in pairs[:10]))
(OUT / "strong_confusion.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
