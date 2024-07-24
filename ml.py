# this script trains and evaluates an XGBoost model for predicting if a peer 2 peer loan will default

# import packages
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import ppscore as pps
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, RocCurveDisplay

# list of columns to be loaded in dataset
usecols = [
    "loan_status",
    "issue_d",
    #'id', # relevancy
    #'member_id', # relevancy
    'loan_amnt',
    #'funded_amnt', # relevancy
    #'funded_amnt_inv', # relevancy
    'term',
    'int_rate',
    'installment',
    #'grade', # relevancy
    'sub_grade',
    #'emp_title', # relevancy
    'emp_length',
    'home_ownership',
    'annual_inc',
    #'pymnt_plan',
    #'url',
    #'desc',
    'purpose',
    #'title',
    #'zip_code',
    #'addr_state',
    'dti',
    'delinq_2yrs',
    'earliest_cr_line',
    'fico_range_low',
    'fico_range_high',
    'inq_last_6mths',
    'mths_since_last_delinq',
    'mths_since_last_record',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',
    'initial_list_status',
    #'last_pymnt_amnt', # leakage
    'last_fico_range_high',
    'last_fico_range_low',
    'collections_12_mths_ex_med',
    'mths_since_last_major_derog',
    'policy_code',
    'annual_inc_joint',
    'dti_joint',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal',
    'open_acc_6m',
    'open_act_il',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'total_rev_hi_lim',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
    'acc_open_past_24mths',
    'avg_cur_bal',
    'bc_open_to_buy',
    'bc_util',
    'chargeoff_within_12_mths',
    'delinq_amnt',
    'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
    'mort_acc',
    'mths_since_recent_bc',
    'mths_since_recent_bc_dlq',
    'mths_since_recent_inq',
    'mths_since_recent_revol_delinq',
    'num_accts_ever_120_pd',
    'num_actv_bc_tl',
    'num_actv_rev_tl',
    'num_bc_sats',
    'num_bc_tl',
    'num_il_tl',
    'num_op_rev_tl',
    'num_rev_accts',
    'num_rev_tl_bal_gt_0',
    'num_sats',
    'num_tl_120dpd_2m',
    'num_tl_30dpd',
    'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m',
    'pct_tl_nvr_dlq',
    'percent_bc_gt_75',
    'pub_rec_bankruptcies',
    'tax_liens',
    'tot_hi_cred_lim',
    'total_bal_ex_mort',
    'total_bc_limit',
    'total_il_high_credit_limit',
    'revol_bal_joint',
    #'sec_app_fico_range_low',
    #'sec_app_fico_range_high',
    #'sec_app_earliest_cr_line',
    #'sec_app_inq_last_6mths',
    #'sec_app_mort_acc',
    #'sec_app_open_acc',
    #'sec_app_revol_util',
    #'sec_app_open_act_il',
    #'sec_app_num_rev_accts',
    #'sec_app_chargeoff_within_12_mths',
    #'sec_app_collections_12_mths_ex_med',
    #'sec_app_mths_since_last_major_derog',
    #'deferral_term', # possible leakage
    #'hardship_amount', # possible leakage
    #'payment_plan_start_date', # possible leakage
    #'orig_projected_additional_accrued_interest', # possible leakage
    #'disbursement_method' # relevancy
    "verification_status"
]

# load data
df = pd.read_csv("C:/Users/gregh/Downloads/archive (6)/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", usecols=usecols)

# filter data
df = df[df["loan_status"] != "Current"] # filter out current loans
df.dropna(subset = ['loan_amnt'], inplace=True)

# feature engineering

# verification status
df["verification_status"] = df["verification_status"].apply(lambda x: 0 if "Not" in x else 1)

# inital list status
df["initial_list_status"] = df["initial_list_status"].apply(lambda x: 0 if x=="w" else 1)

# fico
df["last_fico_range"] = (df["last_fico_range_high"] + df["last_fico_range_low"]) / 2
del df["last_fico_range_high"], df["last_fico_range_low"]

# earliest credit line
def month_difference(date1_str, date2_str):
    try:
        date_format = "%b-%Y"

        # Parse the date strings into datetime objects
        date1 = datetime.strptime(date1_str, date_format)
        date2 = datetime.strptime(date2_str, date_format)

        # Calculate the difference in months
        difference_in_months = (date1.year - date2.year) * 12 + (date1.month - date2.month)

        return difference_in_months
    except:
        return np.nan

months_diffs = []
for issue_d, earliest_cr_line in zip(df["issue_d"].values, df["earliest_cr_line"].values):
    months_diffs.append(month_difference(issue_d, earliest_cr_line))
df["credit_line"] = months_diffs
del df["issue_d"], df["earliest_cr_line"]

# loan term
df["term"] = df["term"].apply(lambda x: int(x[:2]) if str(x) != "nan" else np.nan)

# income
joint_incs = []
for inc, inc_joint in zip(df["annual_inc"].values, df["annual_inc_joint"].values):
  if str(inc_joint) == "nan":
    joint_incs.append(inc)
  else:
    joint_incs.append(inc_joint)
df["income"] = joint_incs
df["inc_to_amnt"] = df["income"] / df["loan_amnt"]
del df["annual_inc"], df["annual_inc_joint"], df["loan_amnt"], df["income"]

# joint balance
joint_bals = []
for bal, bal_joint in zip(df["revol_bal"].values, df["revol_bal_joint"].values):
  if str(bal_joint) == "nan":
    joint_bals.append(bal)
  else:
    joint_bals.append(bal_joint)
df["revolving_balance"] = joint_bals
del df["revol_bal"], df["revol_bal_joint"]

# total revolving high limit
df["total_rev_hi_lim"] = [x if x != 0 else -1 for x in df["total_rev_hi_lim"].fillna(-1)]

# normalize some features by credit
credit_norm_cols = [
    "revol_util",
    "all_util",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "avg_cur_bal",
    "bc_open_to_buy",
    'open_acc',
    'total_acc',
    #"inc_to_amnt",
    "revolving_balance"
]

for col in credit_norm_cols:
    df[col] = df[col] / df["total_rev_hi_lim"]

# employment length    
def emp_length(emp_len):
    try:
        emp_len = emp_len.split()[0]
        if emp_len == "10+":
            emp_len = 10
        elif emp_len == "<":
            emp_len = 0
        else:
            emp_len = int(emp_len)
        return emp_len
    except:
        return np.nan

df["emp_length"] = df["emp_length"].apply(emp_length)

# fico
df["fico"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
del df["fico_range_high"], df["fico_range_low"]

# default
def default(loan_status):
  if loan_status in ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]:
    return 1
  else:
    return 0

df["default"] = df["loan_status"].apply(default)
del df["loan_status"]

# debt to income
joint_dtis = []
for dti, dti_joint in zip(df["dti"].values, df["dti_joint"].values):
  if str(dti_joint) == "nan":
    joint_dtis.append(dti)
  else:
    joint_dtis.append(dti_joint)
df["debt_to_income"] = joint_dtis
del df["dti"], df["dti_joint"]

# grade
d = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

def score_grade(grade):
    try:
        score = d[grade[0]]
        score += (int(grade[1]) - 1) / 5
        return score
    except:
        return np.nan

    df["grade_score"] = df["sub_grade"].apply(score_grade)
del df["sub_grade"]

# treat missing values
#df = df.fillna(-1)

# default rate
print("Default Rate:", df["default"].mean())

# base case
print("Base Case:   ", 1 - df["default"].mean(), '\n')

# prepare data for ml
dfx = df[list(df.columns)]
for col in ["home_ownership", "purpose"]:
    ohe = OneHotEncoder()
    dfx[ohe.categories_[0]] = ohe.fit_transform(dfx[[col]]).toarray()
    del dfx[col]

# predictive power score
dfx["default"] = dfx["default"].astype(str)
pps_df = pps.predictors(dfx.fillna(-1), "default")
plt.figure(figsize=(20, 6))
sns.barplot(data=pps_df, x="x", y="ppscore")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()

# prepare data for ml
dfx["default"] = dfx["default"].astype(int)
y = dfx["default"].values
del dfx["default"]
X = dfx.to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# ml
model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42
)
class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
model.fit(X_train, y_train, sample_weight=class_weights)

# eval
print("Train")
print("Accuracy:", model.score(X_train, y_train))
print("AUC:     ", roc_auc_score(y_train, model.predict(X_train)), "\n")
print("Test")
print("Accuracy:", model.score(X_test, y_test))
print("AUC:     ", roc_auc_score(y_test, model.predict(X_test)), "\n")

# feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': dfx.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(25, 10))
my_plot = sns.barplot(data=feature_importance_df, x="Feature", y="Importance")
my_plot.set_xticklabels(my_plot.get_xticklabels(), rotation=90)
plt.show()

# eval
y_pred = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['fully paid', 'default']); ax.yaxis.set_ticklabels(['fully paid', 'default']);
plt.show()

# default rate of loans in our portfolio
print("Model's Portfolio's Default Rate:               ", cm[1][0] / (cm[1][0] + cm[0][0]))

# accuracy when predicting loan doesn't default
print("Accuracy when model predicts loan won't default:", 1 - cm[1][0] / (cm[1][0] + cm[0][0]), "\n")

# precision, recall, f1-score, support
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
metrics_table = pd.DataFrame({
    'Class': list(range(len(precision))),
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1_score,
    'Support': support
})
average_metrics = pd.DataFrame({
    'Class': ['average'],
    'Precision': [precision.mean()],
    'Recall': [recall.mean()],
    'F1-score': [f1_score.mean()],
    'Support': [support.sum()]
})
metrics_table = pd.concat([metrics_table, average_metrics], ignore_index=True)
print(metrics_table)
del metrics_table["Support"]
metrics_table["Class"] = ["fully paid", "default", "average"]
metrics_table.set_index('Class', inplace=True)
sns.heatmap(metrics_table, annot=True, cmap='YlGnBu', fmt='.3f', cbar=True)
plt.title('Precision, Recall, and F1-score')
plt.show()

# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='Model')
display.plot()
plt.plot([0, 1], [0, 1], linestyle="dashed", color="#1f77b4")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# ml
model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42
)
class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
model.fit(X, y, sample_weight=class_weights)

# deployment
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
