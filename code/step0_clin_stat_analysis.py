# @Function : Clinical statistical analysis: intergroup difference tests (t/U/Chi-square/Fisher) + logistic regression (univariate/multivariate + 95%CI)
# @Renew : Only significant variables are included in multivariate testing
# @Notes: Removed external center variable analysis

import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
from scipy.stats import (ttest_ind, mannwhitneyu,
                         chi2_contingency, fisher_exact,
                         shapiro, levene)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def clinical_statistical_analysis(file_path, label_col, discrete_cols, float_cols):
    """
    Full clinical statistical workflow:
    1. Continuous variables: Normal → t-test; Non-normal → Mann-Whitney U test
    2. Categorical variables: Large sample → Chi-square; Small sample/expected freq<5 → Fisher's exact test
    3. Univariate logistic regression → Select variables with P<0.05 → Multivariate logistic regression (OR+95%CI+P-value)
    """
    # ===================== 1. Data loading and preprocessing =====================
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df = df[df["医院"] != "中山大学附属第三医院"].reset_index(drop=True) # Remove data from The Third Affiliated Hospital of Sun Yat-sen University
    df[label_col] = df[label_col].astype(int)  # Convert label to binary 0/1

    # Combine all columns and filter data
    all_cols = [label_col] + discrete_cols + float_cols
    df = df[all_cols]

    # Missing value imputation: mode for categorical, median for continuous
    df[discrete_cols] = df[discrete_cols].apply(lambda x: x.fillna(x.mode()[0]).astype(int))
    df[float_cols] = df[float_cols].apply(lambda x: x.fillna(x.median()))

    # Grouping: Negative group(0) / Positive group(1)
    group0 = df[df[label_col] == 0]
    group1 = df[df[label_col] == 1]
    print(f"✅ Data grouping completed: Negative group(n={len(group0)}), Positive group(n={len(group1)})")
    print("=" * 120)

    # ===================== 2. Continuous variable intergroup tests: t-test / U-test =====================
    print("【1. Intergroup difference tests for continuous variables】")
    print(f"{'Variable':<20}{'Group description(0/1)':<40}{'Test method':<15}{'Statistic':<10}{'P-value':<10}")
    print("-" * 120)

    for col in float_cols:
        data0 = group0[col].values
        data1 = group1[col].values

        # Normality test (Shapiro-Wilk)
        norm0 = shapiro(data0)[1] > 0.05
        norm1 = shapiro(data1)[1] > 0.05
        # Homogeneity of variance test
        var_eq = levene(data0, data1)[1] > 0.05

        # Clinical description
        desc0 = f"{np.mean(data0):.2f}±{np.std(data0):.2f}" if (norm0 and norm1) else f"{np.median(data0):.2f}({np.percentile(data0,25):.2f}-{np.percentile(data0,75):.2f})"
        desc1 = f"{np.mean(data1):.2f}±{np.std(data1):.2f}" if (norm0 and norm1) else f"{np.median(data1):.2f}({np.percentile(data1,25):.2f}-{np.percentile(data1,75):.2f})"
        desc = f"{desc0} / {desc1}"

        # Select test method
        if norm0 and norm1 and var_eq:
            stat, p_val = ttest_ind(data0, data1)
            method = "t-test"
        else:
            stat, p_val = mannwhitneyu(data0, data1, alternative='two-sided')
            method = "Mann-Whitney U"

        print(f"{col:<20}{desc:<40}{method:<15}{stat:<10.3f}{p_val:<10.3f}")

    print("=" * 120)

    # ===================== 3. Categorical variable intergroup tests: Chi-square / Fisher =====================
    print("【2. Intergroup difference tests for categorical variables】")
    print(f"{'Variable':<25}{'Test method':<18}{'Chi2/U value':<12}{'P-value':<10}")
    print("-" * 120)

    for col in discrete_cols:
        # Build contingency table
        crosstab = pd.crosstab(df[col], df[label_col])
        n_total = crosstab.sum().sum()

        # Calculate expected frequencies to determine test method
        chi2, p_chi2, dof, expected = chi2_contingency(crosstab)
        min_exp = expected.min()

        if n_total >= 40 and min_exp >= 5:
            stat, p_val = chi2, p_chi2
            method = "Chi-square test"
        else:
            # Fisher's exact test (only supports 2×2 tables, multi-category automatically uses Chi-square)
            if crosstab.shape == (2, 2):
                stat, p_val = fisher_exact(crosstab)
                method = "Fisher's exact test"
            else:
                stat, p_val = chi2, p_chi2
                method = "Chi-square test(multi-category)"

        print(f"{col:<25}{method:<18}{stat:<12.3f}{p_val:<10.3f}")

    print("=" * 120)

    # ===================== 4. Univariate logistic regression (OR+95%CI) =====================
    print("【3. Univariate logistic regression analysis (OR+95%CI)】")
    # Dummy encoding + standardization
    X_discrete = pd.get_dummies(df[discrete_cols], drop_first=True)
    X_float = df[float_cols].copy()
    X = pd.concat([X_discrete, X_float], axis=1)
    y = df[label_col]

    # Univariate regression
    single_results = []
    sig_single_vars = []  # Store significant univariate variables (P<0.05)
    print(f"{'Variable':<25}{'OR':<12}{'95% CI':<25}{'P-value':<10}")
    for col in X.columns:
        X_single = sm.add_constant(X[col])
        model = sm.Logit(y, X_single).fit(disp=0)
        coef = model.params[1]
        se = model.bse[1]
        or_val = np.exp(coef)
        ci_l = np.exp(coef - 1.96 * se)
        ci_u = np.exp(coef + 1.96 * se)
        p_val = model.pvalues[1]

        single_results.append((col, or_val, ci_l, ci_u, p_val))
        # Select significant variables
        if p_val < 0.05:
            sig_single_vars.append(col)
        print(f"{col:<25}{or_val:<12.3f}[{ci_l:.3f}, {ci_u:.3f}]    {p_val:<10.3f}")

    # ===================== 5. Multivariate logistic regression (only significant univariate variables) =====================
    print("\n【4. Multivariate logistic regression analysis (OR+95%CI) - Only variables with univariate P<0.05 included】")
    sig_multi_vars = []
    # Key modification: Check if there are significant variables, skip if none
    if not sig_single_vars:
        print("⚠️ No significant univariate variables (P<0.05), skipping multivariate logistic regression analysis")
    else:
        # Build multivariate model using only significant univariate variables
        X_multi = sm.add_constant(X[sig_single_vars])
        model_multi = sm.Logit(y, X_multi).fit(disp=0)
        print(f"{'Variable':<25}{'OR':<12}{'95% CI':<25}{'P-value':<10}")
        for col in model_multi.params.index:
            coef = model_multi.params[col]
            se = model_multi.bse[col]
            or_val = np.exp(coef)
            ci_l = np.exp(coef - 1.96 * se)
            ci_u = np.exp(coef + 1.96 * se)
            p_val = model_multi.pvalues[col]

            if col != 'const':
                print(f"{col:<25}{or_val:<12.3f}[{ci_l:.3f}, {ci_u:.3f}]    {p_val:<10.3f}")
                if p_val < 0.05:
                    sig_multi_vars.append(col)

    # ===================== 6. Significant variable summary =====================
    print("\n" + "=" * 80)
    print(f"📊 Significant univariate variables(P<0.05): {sig_single_vars if sig_single_vars else 'None'}")
    print(f"📊 Significant multivariate variables(P<0.05): {sig_multi_vars if sig_multi_vars else 'None'}")
    return sig_single_vars, sig_multi_vars

# ===================== 【User parameter configuration】 =====================
if __name__ == '__main__':
    # File path
    file_path = "./datasets/Clin/MVI-HCC-samples.xlsx"
    # Outcome label column (binary: 0=negative, 1=positive)
    label_col = "MVI"

    # 🔹 Categorical/discrete variables
    discrete_cols = [
        # Clinical parameters
        "性别（0女；1男）",
        "肝炎类型（0无；1肝炎；2酒精肝）",
        "临床症状（0无；1有）",
        "慢性病史（0无；1有）",
        "原发肿瘤病史",
        "家族恶性肿瘤病史",
        "AFP分类",

        # Ultrasound parameters
        "内部血流信号（报告结果：0无血流；1内部可见血流）",
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
        "背景肝脏（正常0；肝实质回声粗强1；脂肪肝2；肝硬化3）"
    ]

    # 🔹 Continuous variables
    float_cols = [
        # Clinical parameters
        "年龄",
        "CEA",
        "AFP",

        # Ultrasound parameters
        "病灶最大径(单位cm）",
        "动脉期增强开始时间（S）",
        "达峰时间（S）",
        "达峰用时",
        "低增强开始时间(s）",
        "周边浸润（mm)"
    ]

    print(f"Discrete variables: {len(discrete_cols)}, Continuous variables: {len(float_cols)}")

    # Execute analysis
    sig_single, sig_multi = clinical_statistical_analysis(file_path, label_col, discrete_cols, float_cols)