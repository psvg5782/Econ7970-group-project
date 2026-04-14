"""
Member 4 Full Evaluation Script (Fast Version — no SVM retraining)
Bank Telemarketing Prediction - ECON 7970
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

os.makedirs("results/figures/confusion_matrices", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("Loading data...")
X_test_full = pd.read_csv("data/processed/test_features.csv")
y_test_raw  = pd.read_csv("data/processed/test_target.csv")
le = LabelEncoder()
y_test = le.fit_transform(y_test_raw['y'])   # no=0, yes=1

cat_cols = ['job','marital','education','default','housing',
            'loan','contact','poutcome']

def load_scenario(sfile, drop_extra):
    tr = pd.read_csv(f"data/processed/{sfile}")
    y_tr = le.transform(tr['y'])
    # Only drop columns that actually exist
    always_drop = ['y', 'y_num']
    drop_tr  = [c for c in always_drop + drop_extra if c in tr.columns]
    drop_te  = [c for c in always_drop + drop_extra if c in X_test_full.columns]
    tr_after_drop = tr.drop(columns=drop_tr)
    te_after_drop = X_test_full.drop(columns=drop_te)
    tr_enc = pd.get_dummies(tr_after_drop,
                             columns=[c for c in cat_cols if c in tr_after_drop.columns])
    te_enc = pd.get_dummies(te_after_drop,
                             columns=[c for c in cat_cols if c in te_after_drop.columns])
    tr_enc, te_enc = tr_enc.align(te_enc, join='left', axis=1, fill_value=0)
    te_enc = te_enc.reindex(columns=tr_enc.columns, fill_value=0)
    return tr_enc.values.astype(float), y_tr, te_enc.values.astype(float)

S_CONFIGS = {
    'S1': ('train_s1.csv', []),
    'S2': ('train_s2.csv', ['duration']),
    'S3': ('train_s3.csv', ['duration','campaign','pdays','previous','poutcome',
                              'contact','day',
                              'month_aug','month_dec','month_feb','month_jan',
                              'month_jul','month_jun','month_mar','month_may',
                              'month_nov','month_oct','month_sep']),
    'S4': ('train_s4.csv', ['age','job','marital','education','default','balance',
                              'housing','loan','duration','campaign','contact','day',
                              'isretired','isstudent','balance_per_call',
                              'month_aug','month_dec','month_feb','month_jan',
                              'month_jul','month_jun','month_mar','month_may',
                              'month_nov','month_oct','month_sep']),
}
SCENARIO_LABELS = {
    'S1': 'S1: All Features (with duration)',
    'S2': 'S2: Realistic (no duration)',
    'S3': 'S3: Demographics Only',
    'S4': 'S4: Previous Campaign Only',
}

# Only fast models (skip SVM)
FAST_MODELS = {
    'KNN':           KNeighborsClassifier(n_neighbors=15),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                            min_samples_split=5, random_state=42,
                                            n_jobs=-1),
    'XGBoost':       xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=6, use_label_encoder=False,
                                        eval_metric='logloss', random_state=42,
                                        n_jobs=-1),
    'Logistic Reg':  LogisticRegression(max_iter=300, random_state=42, n_jobs=-1),
}

COLORS = {
    'KNN': '#2196F3',
    'Random Forest': '#4CAF50',
    'XGBoost': '#E91E63',
    'Logistic Reg': '#FF9800',
}

print("Training models (KNN, RF, XGBoost, LR — no SVM for speed)...")
results_store = {}

for s_name, (sfile, drop_cols) in S_CONFIGS.items():
    print(f"  Scenario {s_name}...")
    X_tr, y_tr, X_te = load_scenario(sfile, drop_cols)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_sc, y_tr)

    for m_name, model in FAST_MODELS.items():
        for use_smote in [False, True]:
            key = (m_name, s_name, use_smote)
            Xfit = X_tr_sm if use_smote else X_tr_sc
            yfit = y_tr_sm if use_smote else y_tr
            mdl = type(model)(**model.get_params())
            mdl.fit(Xfit, yfit)
            proba = mdl.predict_proba(X_te_sc)[:,1]
            pred  = mdl.predict(X_te_sc)
            results_store[key] = (proba, pred)
    print(f"    done.")

print("All models trained.\n")

# ─────────────────────────────────────────────
# ROC CURVES — 4-panel (one per scenario)
# ─────────────────────────────────────────────
print("Plotting ROC curves...")
for smote_flag, suffix in [(False,''), (True,'_smote')]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    tag = 'With SMOTE' if smote_flag else 'Without SMOTE'
    fig.suptitle(f'ROC Curves — All Models, All Scenarios ({tag})',
                 fontsize=14, fontweight='bold')
    for ax, s_name in zip(axes.flatten(), S_CONFIGS):
        for m_name in FAST_MODELS:
            proba, _ = results_store[(m_name, s_name, smote_flag)]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[m_name], lw=2,
                    label=f'{m_name} (AUC={roc_auc:.3f})')
        ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
        ax.set_title(SCENARIO_LABELS[s_name], fontsize=10)
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/roc_curves{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Individual scenario ROC (solid=no SMOTE, dashed=SMOTE)
for s_name in S_CONFIGS:
    fig, ax = plt.subplots(figsize=(9, 7))
    for m_name in FAST_MODELS:
        proba,   _ = results_store[(m_name, s_name, False)]
        proba_s, _ = results_store[(m_name, s_name, True)]
        fpr, tpr, _ = roc_curve(y_test, proba)
        fpr_s, tpr_s, _ = roc_curve(y_test, proba_s)
        ax.plot(fpr,   tpr,   color=COLORS[m_name], lw=2.5,
                label=f'{m_name} (AUC={auc(fpr,tpr):.3f})')
        ax.plot(fpr_s, tpr_s, color=COLORS[m_name], lw=1.5, linestyle='--', alpha=0.6,
                label=f'{m_name}+SMOTE (AUC={auc(fpr_s,tpr_s):.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_title(f'ROC Curves — {SCENARIO_LABELS[s_name]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11); ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/roc_curves_{s_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
print("  ROC curves done.")

# ─────────────────────────────────────────────
# PR CURVES
# ─────────────────────────────────────────────
print("Plotting PR curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Precision-Recall Curves — All Models, All Scenarios', fontsize=14, fontweight='bold')
for ax, s_name in zip(axes.flatten(), S_CONFIGS):
    for m_name in FAST_MODELS:
        proba, _ = results_store[(m_name, s_name, True)]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=COLORS[m_name], lw=2, label=f'{m_name} AP={ap:.3f}')
    base = y_test.sum()/len(y_test)
    ax.axhline(base, color='gray', linestyle='--', lw=1.2, label=f'Baseline {base:.2f}')
    ax.set_title(SCENARIO_LABELS[s_name], fontsize=10)
    ax.set_xlabel('Recall', fontsize=9); ax.set_ylabel('Precision', fontsize=9)
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/pr_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  PR curves done.")

# ─────────────────────────────────────────────
# CONFUSION MATRICES — Top 3 models S2
# ─────────────────────────────────────────────
print("Confusion matrices...")
s2_f1 = {m: f1_score(y_test, results_store[(m,'S2',True)][1]) for m in FAST_MODELS}
top3  = sorted(s2_f1, key=s2_f1.get, reverse=True)[:3]

for smote_flag, tag in [(False,'no_smote'),(True,'smote')]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Confusion Matrices — Top 3 Models, S2 Realistic'
                 + (' (SMOTE)' if smote_flag else ''), fontsize=13, fontweight='bold')
    for ax, m_name in zip(axes, top3):
        _, pred = results_store[(m_name,'S2',smote_flag)]
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No','Yes'], yticklabels=['No','Yes'],
                    annot_kws={'size':14})
        p = precision_score(y_test,pred); r = recall_score(y_test,pred); f = f1_score(y_test,pred)
        ax.set_title(f'{m_name}\nP={p:.3f}  R={r:.3f}  F1={f:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('Actual', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"results/figures/confusion_matrices/cm_top3_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close()
print("  Confusion matrices done.")

# ─────────────────────────────────────────────
# DURATION INFLATION
# ─────────────────────────────────────────────
print("Duration inflation...")
infl_rows = []
for m in FAST_MODELS:
    for sf in [False, True]:
        a1 = roc_auc_score(y_test, results_store[(m,'S1',sf)][0])
        a2 = roc_auc_score(y_test, results_store[(m,'S2',sf)][0])
        infl_rows.append({'Model':m,'SMOTE':'Yes' if sf else 'No',
                          'AUC_S1':round(a1,4),'AUC_S2':round(a2,4),
                          'Duration_Inflation_%':round((a1-a2)/a2*100,2)})
df_infl = pd.DataFrame(infl_rows)
df_infl.to_csv("results/tables/duration_inflation.csv", index=False)

mlist = list(FAST_MODELS.keys())
x = np.arange(len(mlist)); w = 0.35
v_no  = [df_infl[(df_infl.Model==m)&(df_infl.SMOTE=='No')]['Duration_Inflation_%'].values[0] for m in mlist]
v_yes = [df_infl[(df_infl.Model==m)&(df_infl.SMOTE=='Yes')]['Duration_Inflation_%'].values[0] for m in mlist]

fig, ax = plt.subplots(figsize=(10,6))
b1 = ax.bar(x-w/2, v_no,  w, label='No SMOTE',   color='#2196F3', alpha=0.85)
b2 = ax.bar(x+w/2, v_yes, w, label='With SMOTE', color='#FF9800', alpha=0.85)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x()+bar.get_width()/2,h),
                xytext=(0,3), textcoords='offset points', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(mlist, fontsize=11)
ax.set_ylabel('Duration Inflation (%)', fontsize=12)
ax.set_title('Duration Inflation: S1 vs S2 — How much does "duration" inflate ROC-AUC?',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.axhline(0, color='k', lw=0.8); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/duration_inflation.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Duration inflation done.")

# ─────────────────────────────────────────────
# SCENARIO COMPARISON (S2 vs S3 vs S4)
# ─────────────────────────────────────────────
print("Scenario comparison...")
sc_rows = []
for m in FAST_MODELS:
    for s in ['S2','S3','S4']:
        for sf in [False, True]:
            proba, pred = results_store[(m,s,sf)]
            sc_rows.append({'Model':m,'Scenario':s,'SMOTE':'Yes' if sf else 'No',
                            'ROC-AUC': round(roc_auc_score(y_test,proba),4),
                            'Recall':  round(recall_score(y_test,pred),4),
                            'F1':      round(f1_score(y_test,pred),4)})
df_sc = pd.DataFrame(sc_rows)

fig, axes = plt.subplots(1, 2, figsize=(16,6))
for ax, metric in zip(axes, ['ROC-AUC','Recall']):
    piv = df_sc[df_sc.SMOTE=='Yes'].pivot(index='Model', columns='Scenario', values=metric)[['S2','S3','S4']]
    xp  = np.arange(len(piv)); ww = 0.25
    for i,(col,col_) in enumerate(zip(['S2','S3','S4'],['#2196F3','#4CAF50','#FF9800'])):
        brs = ax.bar(xp+i*ww, piv[col], ww, label=col, color=col_, alpha=0.85)
        for b in brs:
            h=b.get_height()
            ax.annotate(f'{h:.2f}', xy=(b.get_x()+b.get_width()/2,h),
                        xytext=(0,2), textcoords='offset points', ha='center', fontsize=7)
    ax.set_xticks(xp+ww); ax.set_xticklabels(piv.index, fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Scenario (With SMOTE)', fontsize=12, fontweight='bold')
    ax.legend(title='Scenario', fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.suptitle('Scenario Comparison: Realistic vs Demographics vs Previous Campaign',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("results/figures/scenario_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Scenario comparison done.")

# ─────────────────────────────────────────────
# SMOTE IMPACT
# ─────────────────────────────────────────────
print("SMOTE impact...")
fig, ax = plt.subplots(figsize=(12,6))
mlist2 = list(FAST_MODELS.keys())
xp = np.arange(len(mlist2)); ww = 0.2
pal = ['#2196F3','#4CAF50']
for si, s_name in enumerate(['S1','S2']):
    r_no  = [recall_score(y_test, results_store[(m,s_name,False)][1]) for m in mlist2]
    r_yes = [recall_score(y_test, results_store[(m,s_name,True)][1])  for m in mlist2]
    offset_no  = xp + (si*2-1.5)*ww
    offset_yes = xp + (si*2-0.5)*ww
    ax.bar(offset_no,  r_no,  ww, label=f'{s_name} No SMOTE',   color=pal[si], alpha=0.6)
    ax.bar(offset_yes, r_yes, ww, label=f'{s_name} With SMOTE', color=pal[si], alpha=1.0, hatch='//')
ax.set_xticks(xp); ax.set_xticklabels(mlist2, fontsize=11)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('SMOTE Impact on Recall — S1 and S2', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0,1.1)
plt.tight_layout()
plt.savefig("results/figures/smote_impact.png", dpi=150, bbox_inches='tight')
plt.close()
print("  SMOTE impact done.")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (RF + XGBoost on S2)
# ─────────────────────────────────────────────
print("Feature importance...")
tr_s2 = pd.read_csv("data/processed/train_s2.csv")
drop_s2 = [c for c in ['duration','y','y_num'] if c in tr_s2.columns]
tr_enc2 = pd.get_dummies(tr_s2.drop(columns=drop_s2),
                          columns=[c for c in cat_cols if c in tr_s2.columns])
feat_names = tr_enc2.columns.tolist()
X_tr2 = tr_enc2.values.astype(float)
y_tr2 = le.transform(tr_s2['y'])
sc2 = StandardScaler(); X_tr2_sc = sc2.fit_transform(X_tr2)

sm2 = SMOTE(random_state=42); X_sm2, y_sm2 = sm2.fit_resample(X_tr2_sc, y_tr2)

rf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf2.fit(X_sm2, y_sm2)
xgb2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                           use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb2.fit(X_sm2, y_sm2)

rf_imp  = pd.Series(rf2.feature_importances_,  index=feat_names)
xgb_imp = pd.Series(xgb2.feature_importances_, index=feat_names)
comb    = ((rf_imp + xgb_imp)/2).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10,7))
colors  = ['#E91E63' if i<3 else '#2196F3' for i in range(len(comb))]
comb.sort_values().plot.barh(ax=ax, color=colors[::-1], alpha=0.85)
ax.set_xlabel('Average Importance (RF + XGBoost)', fontsize=12)
ax.set_title('Top 15 Features — Combined Importance\n(RF + XGBoost, S2 Realistic Scenario, with SMOTE)',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/feature_importance_combined.png", dpi=150, bbox_inches='tight')
plt.close()

# Save importance table
comb_df = pd.DataFrame({'Feature': comb.index, 'Importance': comb.values})
comb_df.to_csv("results/tables/feature_importance_combined.csv", index=False)
print("  Feature importance done.")

# ─────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
print("Threshold optimization...")
best_m = max(FAST_MODELS.keys(), key=lambda m: roc_auc_score(y_test, results_store[(m,'S2',True)][0]))
best_proba_sm = results_store[(best_m,'S2',True)][0]

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, best_proba_sm)
f1_arr = np.where((prec_arr+rec_arr)==0, 0, 2*prec_arr*rec_arr/(prec_arr+rec_arr))
best_idx = np.argmax(f1_arr)
best_thr = thr_arr[best_idx]
best_f1  = f1_arr[best_idx]
best_p   = prec_arr[best_idx]
best_r   = rec_arr[best_idx]

print(f"  Best model: {best_m} | Optimal threshold: {best_thr:.3f}")
print(f"  At this threshold → Precision={best_p:.3f}, Recall={best_r:.3f}, F1={best_f1:.3f}")

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(thr_arr, f1_arr[:-1],   label='F1 Score',  color='#E91E63', lw=2)
ax.plot(thr_arr, prec_arr[:-1], label='Precision', color='#2196F3', lw=2)
ax.plot(thr_arr, rec_arr[:-1],  label='Recall',    color='#4CAF50', lw=2)
ax.axvline(best_thr, color='gray', linestyle='--', lw=1.5,
           label=f'Optimal = {best_thr:.3f}')
ax.scatter([best_thr],[best_f1], color='#E91E63', s=100, zorder=5)
ax.set_xlabel('Decision Threshold', fontsize=12); ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Threshold Optimization — {best_m} + SMOTE (S2 Realistic)\n'
             f'Optimal threshold = {best_thr:.3f}  →  Recall={best_r:.3f}, F1={best_f1:.3f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/threshold_optimization.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Threshold optimization done.")

# ─────────────────────────────────────────────
# BUSINESS RECOMMENDATION — Cumulative Gains
# ─────────────────────────────────────────────
print("Business recommendation...")
df_biz = pd.DataFrame({'prob':best_proba_sm,'actual':y_test})
df_biz = df_biz.sort_values('prob',ascending=False).reset_index(drop=True)
total_buyers = df_biz['actual'].sum()
cum_buyers   = df_biz['actual'].cumsum()
cum_pct_called = (np.arange(1,len(df_biz)+1))/len(df_biz)*100
cum_pct_buyers = cum_buyers/total_buyers*100

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cum_pct_called, cum_pct_buyers, color='#E91E63', lw=2.5, label='Model Lift Curve')
ax.plot([0,100],[0,100],'k--', lw=1.5, alpha=0.5, label='Random Baseline')
for pct in [10,20,30]:
    idx = int(pct/100*len(df_biz))-1
    bc  = cum_pct_buyers.iloc[idx]
    ax.scatter([pct],[bc], s=80, zorder=5, color='#E91E63')
    ax.annotate(f'Top {pct}%\n→ {bc:.0f}% buyers', xy=(pct,bc),
                xytext=(pct+4, bc-10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('% of Customers Called', fontsize=12)
ax.set_ylabel('% of Actual Buyers Reached', fontsize=12)
ax.set_title(f'Cumulative Gains Curve — {best_m} + SMOTE (S2 Realistic)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/business_recommendation.png", dpi=150, bbox_inches='tight')
plt.close()

biz_rows = []
for pct in [10,20,30,40,50]:
    idx = int(pct/100*len(df_biz))-1
    bc  = cum_pct_buyers.iloc[idx]
    biz_rows.append({'Top_X_Pct_Called':f'{pct}%',
                     'Customers_Called':int(pct/100*len(df_biz)),
                     'Buyers_Caught_Pct':f'{bc:.1f}%',
                     'Calls_Saved_Pct':f'{100-pct}%',
                     'Lift':f'{bc/pct:.2f}x'})
pd.DataFrame(biz_rows).to_csv("results/tables/business_recommendation.csv", index=False)
print("  Business recommendation done.")
print(pd.DataFrame(biz_rows).to_string(index=False))

# ─────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\nBuilding final_comparison.csv...")
our_rows = []
for m in FAST_MODELS:
    for s in ['S1','S2','S3','S4']:
        for sf in [False, True]:
            proba, pred = results_store[(m,s,sf)]
            our_rows.append({'Member':'M3-Advanced','Model':m,'Scenario':s,
                             'SMOTE':'Yes' if sf else 'No',
                             'Accuracy': round(accuracy_score(y_test,pred),4),
                             'Precision':round(precision_score(y_test,pred),4),
                             'Recall':   round(recall_score(y_test,pred),4),
                             'F1':       round(f1_score(y_test,pred),4),
                             'ROC-AUC':  round(roc_auc_score(y_test,proba),4)})

df_our = pd.DataFrame(our_rows)
df_m2  = pd.read_csv("results/tables/baseline_results_original_features.csv")
df_m2['Member'] = 'M2-Baseline'

cols = ['Member','Model','Scenario','SMOTE','Accuracy','Precision','Recall','F1','ROC-AUC']
df_final = pd.concat([df_m2[cols], df_our[cols]], ignore_index=True)
df_final = df_final.sort_values(['Scenario','SMOTE','ROC-AUC'], ascending=[True,True,False])
df_final['Rank'] = df_final.groupby(['Scenario','SMOTE'])['ROC-AUC'].rank(ascending=False, method='first').astype(int)
df_final.to_csv("results/tables/final_comparison.csv", index=False)
print(f"  final_comparison.csv — {len(df_final)} rows")

# Summary: best model per scenario
print("\n=== Best model per scenario (ROC-AUC, with SMOTE) ===")
best = df_final[df_final.SMOTE=='Yes'].sort_values('ROC-AUC',ascending=False).groupby('Scenario').first()
print(best[['Model','Accuracy','Recall','F1','ROC-AUC']].to_string())

# ─────────────────────────────────────────────
# INTERPRETABILITY TABLE
# ─────────────────────────────────────────────
df_lr = df_m2[(df_m2.Model=='Logistic Regression')&(df_m2['Scenario'].str.contains('S2'))]
lr_no  = df_lr[df_lr.SMOTE=='No'][['ROC-AUC','Recall','F1']].values[0]
lr_yes = df_lr[df_lr.SMOTE=='Yes'][['ROC-AUC','Recall','F1']].values[0]

proba_xn, pred_xn = results_store[('XGBoost','S2',False)]
proba_xy, pred_xy = results_store[('XGBoost','S2',True)]

interp = pd.DataFrame([
    {'Model':'Logistic Regression','Interpretable':'Yes (coefficients visible)',
     'ROC-AUC_NoSMOTE':lr_no[0],'Recall_NoSMOTE':lr_no[1],'F1_NoSMOTE':lr_no[2],
     'ROC-AUC_SMOTE':lr_yes[0], 'Recall_SMOTE':lr_yes[1], 'F1_SMOTE':lr_yes[2]},
    {'Model':'XGBoost','Interpretable':'No (Black Box)',
     'ROC-AUC_NoSMOTE':round(roc_auc_score(y_test,proba_xn),4),
     'Recall_NoSMOTE': round(recall_score(y_test,pred_xn),4),
     'F1_NoSMOTE':     round(f1_score(y_test,pred_xn),4),
     'ROC-AUC_SMOTE':  round(roc_auc_score(y_test,proba_xy),4),
     'Recall_SMOTE':   round(recall_score(y_test,pred_xy),4),
     'F1_SMOTE':       round(f1_score(y_test,pred_xy),4)},
])
interp.to_csv("results/tables/interpretability_tradeoff.csv", index=False)
print("\n=== Interpretability Trade-off ===")
print(interp[['Model','Interpretable','ROC-AUC_NoSMOTE','ROC-AUC_SMOTE','Recall_SMOTE']].to_string(index=False))

print("\n✅  All outputs generated!")
print("\nFigures:")
for f in sorted(os.listdir("results/figures")):
    if os.path.isfile(f"results/figures/{f}"):
        print(f"  results/figures/{f}")
for f in sorted(os.listdir("results/figures/confusion_matrices")):
    print(f"  results/figures/confusion_matrices/{f}")
print("\nTables:")
for f in sorted(os.listdir("results/tables")):
    print(f"  results/tables/{f}")

# Store key numbers for report
report_data = {
    'best_model': best_m,
    'best_threshold': round(best_thr, 3),
    'best_recall': round(best_r, 3),
    'best_f1': round(best_f1, 3),
    'best_precision': round(best_p, 3),
}
print("\n=== Key Numbers for Report ===")
for k,v in report_data.items():
    print(f"  {k}: {v}")
