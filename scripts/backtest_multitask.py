"""
Backtest multi-task baselines (winner/method/round) using gold features + labels from golden CSV.
Time-series CV with n_splits=5; writes per-fold metrics to outputs.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.models.calibration_multiclass import WinnerPlatt, TemperatureScaling


def load_meta_X_y():
    gold = pd.read_parquet(ROOT / 'data' / 'gold_features.parquet')
    labels_path = ROOT.parents[0] / 'FightIQ' / 'data' / 'UFC_full_data_golden.csv'
    lab = pd.read_csv(labels_path)
    gold['fight_url'] = gold['fight_url'].astype(str).str.strip().str.rstrip('/')
    lab['fight_url'] = lab['fight_url'].astype(str).str.strip().str.rstrip('/')
    cols = ['fight_url','winner_encoded','result','finish_round','event_date']
    lab = lab[[c for c in cols if c in lab.columns]].drop_duplicates(subset=['fight_url'])
    df = gold.merge(lab, on='fight_url', how='inner', suffixes=('_gold','_lab'))
    # ensure event_date from gold if present
    if 'event_date_gold' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date_gold'], errors='coerce')
    elif 'event_date_lab' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date_lab'], errors='coerce')
    else:
        df['event_date'] = pd.NaT
    # Targets
    y_win = df['winner_encoded'] if 'winner_encoded' in df.columns else None
    y_met = None
    if 'result' in df.columns:
        def map_method(x: str):
            x = str(x).lower()
            if any(k in x for k in ['ko','tko']):
                return 'KO_TKO'
            if 'sub' in x:
                return 'SUB'
            if 'dec' in x or 'decision' in x:
                return 'DEC'
            return 'OTHER'
        y_met = df['result'].map(map_method)
    y_rnd = df['finish_round'] if 'finish_round' in df.columns else None
    # Features
    drop_cols = [c for c in ['fight_url','event_url','event_name','event_date','winner_encoded','result','finish_round'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    meta = df[['fight_url','event_date']]
    return meta, X, y_win, y_met, y_rnd


def time_splits(meta: pd.DataFrame, n_splits=5):
    order = meta.sort_values('event_date').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(order):
        yield order.loc[tr, 'fight_url'].values, order.loc[va, 'fight_url'].values


def evaluate_task(meta, X, y, task: str):
    if y is None or y.isna().all():
        return []
    rows = []
    # Prepare
    mask = pd.notna(y)
    Xf, yf, mf = X[mask], y[mask], meta[mask]
    classes = None
    params = {
        'learning_rate': 0.07,
        'num_leaves': 31,
        'verbose': -1
    }
    if task == 'winner':
        params.update({'objective':'binary', 'metric':'binary_logloss'})
    else:
        if task == 'method':
            classes = sorted(yf.unique().tolist())
        else:
            classes = sorted(yf.dropna().unique().astype(int).tolist())
        params.update({'objective':'multiclass', 'metric':'multi_logloss', 'num_class': len(classes)})

    for i, (tr_ids, va_ids) in enumerate(time_splits(mf, n_splits=5), start=1):
        tr_mask = mf['fight_url'].isin(tr_ids)
        va_mask = mf['fight_url'].isin(va_ids)
        Xtr_full, ytr_full = Xf[tr_mask], yf[tr_mask]
        Xva, yva = Xf[va_mask], yf[va_mask]
        # inner calibration split on training fold (last 10%)
        tr_order = mf[tr_mask].sort_values('event_date').reset_index(drop=True)
        split_idx = int(len(tr_order) * 0.9)
        cal_ids = tr_order.loc[split_idx:, 'fight_url']
        cal_mask = mf['fight_url'].isin(cal_ids) & tr_mask
        tr_inner_mask = tr_mask & (~cal_mask)
        Xtr, ytr = Xf[tr_inner_mask], yf[tr_inner_mask]
        Xcal, ycal = Xf[cal_mask], yf[cal_mask]

        if task == 'winner':
            mdl = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=300, callbacks=[lgb.log_evaluation(period=0)])
            p_cal = mdl.predict(Xcal)
            cal = WinnerPlatt().fit(p_cal, ycal.values)
            p = cal.transform(mdl.predict(Xva))
            # metrics
            rows.append({'fold': i, 'task': task, 'logloss': float(log_loss(yva, p, labels=[0,1])), 'acc': float(accuracy_score(yva, (p>=0.5).astype(int)))})
        else:
            ytr_c = pd.Categorical(ytr, categories=classes).codes
            mdl = lgb.train(params, lgb.Dataset(Xtr, label=ytr_c), num_boost_round=300, callbacks=[lgb.log_evaluation(period=0)])
            # temperature scaling on calibration slice
            p_cal = mdl.predict(Xcal)
            logits = np.log(np.clip(p_cal, 1e-12, 1.0))
            ycal_c = pd.Categorical(ycal, categories=classes).codes
            temp = TemperatureScaling().fit(logits, ycal_c)
            p = temp.transform(np.log(np.clip(mdl.predict(Xva), 1e-12, 1.0)))
            yva_c = pd.Categorical(yva, categories=classes).codes
            rows.append({'fold': i, 'task': task, 'logloss': float(log_loss(yva_c, p, labels=list(range(len(classes))))), 'acc': float(accuracy_score(yva_c, p.argmax(axis=1)))})
    return rows


def main():
    cfg = load_config(ROOT / 'config/config.yaml')
    logger.remove()
    logger.add(sys.stderr, level=cfg.get('logging',{}).get('level','INFO'))

    meta, X, y_win, y_met, y_rnd = load_meta_X_y()
    records = []
    records += evaluate_task(meta, X, y_win, 'winner')
    records += evaluate_task(meta, X, y_met, 'method')
    records += evaluate_task(meta, X, y_rnd, 'round')
    if not records:
        logger.warning('No records generated')
        return
    df = pd.DataFrame(records)
    out = ROOT / 'outputs' / 'backtest_multitask_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote multi-task backtest summary to {out}")


if __name__ == '__main__':
    main()
