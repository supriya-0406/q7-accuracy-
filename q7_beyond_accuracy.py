"""
🎯 Tophawks Q7: Beyond Accuracy — Metric Selection for Imbalanced Lead Data
Demonstrates why accuracy fails, and how Precision/Recall/F1 align with sales goals.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# 1. Simulate realistic B2B lead data (1,000 leads, 15% conversion rate)
np.random.seed(42)
n_leads = 1000
y_true = np.random.binomial(1, p=0.15, size=n_leads)  # 1 = converted, 0 = lost

# 2. Model A: Naive Baseline (Always predicts "Will NOT convert")
y_pred_naive = np.zeros(n_leads, dtype=int)

# 3. Model B: Realistic ML Lead Scorer
# Catches 80% of converters, but mislabels 20% of non-converters as "hot"
y_pred_ml = y_true.copy()
# False Negatives: Miss 20% of real converters
fn_idx = np.random.choice(np.where(y_true == 1)[0], size=30, replace=False)
y_pred_ml[fn_idx] = 0
# False Positives: Flag 20% of non-converters as high intent
fp_idx = np.random.choice(np.where(y_true == 0)[0], size=170, replace=False)
y_pred_ml[fp_idx] = 1

# 4. Calculate & Print Metrics
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n📊 {name}")
    print(f"   Accuracy:  {acc:.2%}  ← {'MISLEADING' if acc > 0.8 else 'OK'}")
    print(f"   Precision: {prec:.2%}  ← 'Of leads we called, how many actually converted?'" )
    print(f"   Recall:    {rec:.2%}  ← 'Of all converters, how many did we catch?'" )
    print(f"   F1-Score:  {f1:.2%}  ← Harmonic balance of Precision & Recall" )
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} | FN={cm[1,0]} TP={cm[1,1]}")

print("🔍 WHY ACCURACY IS MISLEADING FOR LEAD SCORING (15% Baseline Conversion)")
print("="*65)
evaluate_model("Naive Model (Always predicts 'NO')", y_true, y_pred_naive)
evaluate_model("ML Lead Scorer", y_true, y_pred_ml)

# 5. Business Threshold Tuning (What Sales Actually Needs)
print("\n💡 SALES-OPTIMIZED THRESHOLD STRATEGY:")
print("   → Goal: Maximize Recall (catch converters) while keeping Precision ≥ 60%")
print("   → Action: Score all leads, route top 30% to AEs, rest to nurture.")
print("   → Metric to Track: Recall@Top30% & Precision@HighScore")
