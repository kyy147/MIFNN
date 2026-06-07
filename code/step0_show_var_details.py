import os
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd

def _format_continuous(series: pd.Series) -> str:
    """Continuous variable: mean (min-max)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return "NA(NA-NA)"
    mean = float(s.mean())
    mn = float(s.min())
    mx = float(s.max())
    return f"{mean:.1f}({mn:.1f}-{mx:.1f})"


def _format_discrete(series: pd.Series) -> str:
    """
    Discrete variable: 0: x count (percentage), 1: y count (percentage)...
    Percentage is based on non-null sample count n_valid
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    n = len(s)
    if n == 0:
        return "NA"

    counts = s.value_counts().sort_index()
    parts = []
    for k, v in counts.items():
        pct = v / n * 100.0
        parts.append(f"{int(k)}: {int(v)}({pct:.1f}%)")
    return "\n".join(parts)


def _load_pt_keys(pt_path: str) -> set:
    """
    Load pt file keys and filter out keys not present in internal pt file
    """
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        all_keys = set(map(str, data.keys()))
        
        internal_pt_path = './datasets/pt_file/all_patient_labels.pt'
        internal_data = torch.load(internal_pt_path, map_location='cpu', weights_only=False)
        internal_keys = set(map(str, internal_data.keys()))
        
        return all_keys.intersection(internal_keys)
        
    except Exception as e:
        print(f"Error loading pt files: {e}")
        return set()


def _print_split_summary(title: str, sub: pd.DataFrame, split_keys_count: int,
                         df_total_count: int, id_col: str,
                         discrete_cols: list, float_cols: list, label_col: str = None):
    print("\n" + "=" * 80)
    print(title)
    print(f"Matched samples: {len(sub)} / split keys: {split_keys_count} / total xlsx: {df_total_count}")

    if label_col and label_col in sub.columns:
        print("\nLabel distribution:")
        print(label_col)
        print(_format_discrete(sub[label_col]))

    print("\n--- Discrete variables ---")
    for col in discrete_cols:
        if col not in sub.columns:
            continue
        print(col)
        print(_format_discrete(sub[col]))

    print("\n--- Continuous variables ---")
    for col in float_cols:
        if col not in sub.columns:
            continue
        print(col)
        print(_format_continuous(sub[col]))


def summarize_variables_all_trainval_test(
    xlsx_path: str,
    id_col: str,
    train_pt_path: str,
    val_pt_path: str,
    test_pt_path: str,
    discrete_cols: list,
    float_cols: list,
    label_col: str = None,
):
    """
    Output three sections: ALL, Train+Val, Test
    ALL uses train∪val∪test keys to filter xlsx
    """

    for p in [train_pt_path, val_pt_path, test_pt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"PT file not found: {p}")

    df = pd.read_excel(xlsx_path, engine='openpyxl')
    df[id_col] = df[id_col].astype(str)

    want_cols = [id_col] + (discrete_cols or []) + (float_cols or [])
    if label_col:
        want_cols.append(label_col)
    want_cols = list(dict.fromkeys(want_cols))

    missing = [c for c in want_cols if c not in df.columns]
    if missing:
        print("⚠️ Missing columns in xlsx (automatically ignored):")
        for c in missing:
            print(" -", c)
    keep_cols = [c for c in want_cols if c in df.columns]
    df = df[keep_cols].copy()

    train_ids = _load_pt_keys(train_pt_path)
    val_ids = _load_pt_keys(val_pt_path)
    test_ids = _load_pt_keys(test_pt_path)

    trainval_ids = train_ids | val_ids
    all_ids = train_ids | val_ids | test_ids

    df_all = df[df[id_col].isin(all_ids)].copy()
    df_trainval = df[df[id_col].isin(trainval_ids)].copy()
    df_test = df[df[id_col].isin(test_ids)].copy()

    print(f"XLSX: {xlsx_path}")
    print(f"train: {train_pt_path}")
    print(f"val  : {val_pt_path}")
    print(f"test : {test_pt_path}")

    _print_split_summary(
        title="[ALL] Variable summary (train ∪ val ∪ test)",
        sub=df_all,
        split_keys_count=len(all_ids),
        df_total_count=len(df),
        id_col=id_col,
        discrete_cols=discrete_cols,
        float_cols=float_cols,
        label_col=label_col
    )

    _print_split_summary(
        title="[TRAIN + VAL] Variable summary (train ∪ val)",
        sub=df_trainval,
        split_keys_count=len(trainval_ids),
        df_total_count=len(df),
        id_col=id_col,
        discrete_cols=discrete_cols,
        float_cols=float_cols,
        label_col=label_col
    )

    _print_split_summary(
        title="[TEST] Variable summary",
        sub=df_test,
        split_keys_count=len(test_ids),
        df_total_count=len(df),
        id_col=id_col,
        discrete_cols=discrete_cols,
        float_cols=float_cols,
        label_col=label_col
    )


if __name__ == "__main__":
    # ====== Please change the path to yours. ======
    xlsx_path = "./datasets/Clin/MVI-HCC-samples.xlsx"
    train_pt_path = "./datasets/pt_file/train_fold1.pt"
    val_pt_path   = "./datasets/pt_file/internal_test_fold1.pt"
    test_pt_path  = "./datasets/pt_file/external_test_fold1.pt"

    id_col = "患者编号"
    label_col = "标签1（MVI+卫星灶；0阴性；阳性1）"

    discrete_cols = [
        "肝炎类型（0无；1肝炎；2酒精肝）",
        "临床症状（0无；1有）",
        "内部血流信号（报告结果：0无血流；1内部可见血流）",
        "慢性病史（0无；1有）",
        "病灶形态（报告结果，圆形0；椭圆形1；分叶状2；不规则形3；其他4）",
        "病灶边界(报告结果，清0；不清1)",
        "病灶内部回声水平（报告结果；其他0；低回声1）",
        "病灶内部回声分布（均匀0；不均匀1）",
        "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）",
        "门脉期（0：等增强；1低增强；2高增强）",
        "延迟期(等0；1低)",
        "边界光整（0光整；1不光整）",
        "后方回声（0正常；1增强；2衰减）",
        "多发融合结节（0无；1有）",
        "周边低回声晕（0无；1有）",
        "假包膜（增强影像：0无；1有）",
        "坏死（0无；1有）",
        "CDFI自评",
        "原发肿瘤病史",
        "家族恶性肿瘤病史",
        "背景肝脏（正常0；肝实质回声粗强1；脂肪肝2；肝硬化3）",
        "AFP分类",
        "性别（0女；1男）"
    ]

    float_cols = [
        "病灶最大径(单位cm）",
        "动脉期增强开始时间（S）",
        "达峰时间（S）",
        "达峰用时",
        "低增强开始时间(s）",
        "年龄",
        "CEA",
        "AFP",
        "周边浸润（mm)"
    ]

    summarize_variables_all_trainval_test(
        xlsx_path=xlsx_path,
        id_col=id_col,
        train_pt_path=train_pt_path,
        val_pt_path=val_pt_path,
        test_pt_path=test_pt_path,
        discrete_cols=discrete_cols,
        float_cols=float_cols,
        label_col=label_col,
    )