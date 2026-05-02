
## 📄 Q7: `README.md` (Beyond Accuracy)
```markdown
# 🎯 Tophawks Q7: Metric Selection for Imbalanced Lead Data

Demonstrates why **accuracy is misleading** for lead scoring (15% baseline conversion) and how Precision/Recall/F1 align with sales workflow efficiency.

## ✨ What It Proves
- A naive model that predicts "NO" for all leads achieves **85% accuracy** but **0% recall** → misses every deal
- Realistic ML scorer: 78% accuracy, 80% recall, 38% precision → catches converters but needs threshold tuning
- **Sales-Optimized Metric**: `Recall@TopK` + `Precision@HighScore` > overall accuracy

## 🚀 How to Run
```bash
pip install scikit-learn pandas numpy
python q7_beyond_accuracy.py
