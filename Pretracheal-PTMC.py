# %% [markdown]
# 新的代码仓库总侧区2024.11.24
# 

# %%
pip install pandas scikit-learn matplotlib
#安装包
!pip install -U scikit-learn
##安装pandas的包用于数据的读取与纳入
!pip install pandas


# %% [markdown]
# 导入数据-----

# %%
#2、数据规范化
#指定pandas为pd方便后续数据的读取
import pandas as pd

# %%

data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/3.V2p.csv')
data.head

# %% [markdown]
# 1、分类变量的编码

# %%
#1、分类变量的编码
data.head


# %%
#1.1找出分类型的变量
data_category = data.select_dtypes(include=['object'])

# %%
#1.2查看
data_category

# %%
#1.3把分类之后的变量与之前的拼接在一起,这一步是在找剩下了的数值型变量
data_Number=data.select_dtypes(exclude=['object'])

# %%
#1.4查看数值型的变量有哪些
data_Number

# %%
#1.5看看数值型的变量名字
data_Number.columns.values

# %%
#1.6整合编码
from sklearn.preprocessing import OrdinalEncoder

# 创建并拟合编码器
encoder = OrdinalEncoder()
encoder.fit(data_category)

# 将分类变量进行编码转换
data_category_enc = pd.DataFrame(encoder.transform(data_category), columns=data_category.columns)



# %%
#1.7加载表头
data_category_enc

# %%
#1.8查看某一变量是否正确
data_category_enc['Age'].value_counts()

# %%
#1.9看之前的变量名字
data_category['Age'].value_counts()

# %%
#1.10将表格拼回去
data_enc=pd.concat([data_category_enc,data_Number],axis=1)
#axis=0为纵向拼接 axis=1是按列拼接

# %%
#1.11编码完成
data_enc

# %%
#1.12将新的编码后的数据输入文件夹中
data_enc.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/3.V2T编码后.csv')

# %% [markdown]
# 2、缺失值插补--多重插补法

# %%
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

# 读取数据/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/2.V1编码后.csv
data_enc = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/2.V1T编码后.csv')

# 将分类变量转换为数值类型
categorical_cols = ['Prelaryngeal LNM', 'Pretracheal LNM', 'Paratracheal LNM', 'Con-Paratracheal LNM', 'LNM-prRLN','Sex']
for col in categorical_cols:
    data_enc[col] = data_enc[col].astype('category').cat.codes

# 定义所有包含缺失值的变量
all_cols_with_na = categorical_cols + ['Prelaryngeal LNMR', 'Pretracheal LNMR', 'Paratracheal LNMR', 'Con-Paratracheal LNMR', 'LNMR-prRLN', 
                                       'Prelaryngeal NLNM', 'Pretracheal NLNM', 'Paratracheal NLNM', 'Con-Paratracheal NLNM', 'NLNM-prRLN']

# 使用IterativeImputer进行插补
imputer = IterativeImputer(random_state=0)
data_enc[all_cols_with_na] = imputer.fit_transform(data_enc[all_cols_with_na])

# 确保分类变量的值在合理范围内，并转换回类别类型
for col in categorical_cols:
    data_enc[col] = data_enc[col].round().astype('int').clip(lower=0, upper=data_enc[col].nunique() - 1).astype('category')

# 确保比例变量在0到1之间
ratio_cols = ['Prelaryngeal LNMR', 'Pretracheal LNMR', 'Paratracheal LNMR', 'Con-Paratracheal LNMR', 'LNMR-prRLN']
data_enc[ratio_cols] = data_enc[ratio_cols].clip(lower=0, upper=1)

# 确保计数变量为非负整数
count_cols = ['Prelaryngeal NLNM', 'Pretracheal NLNM', 'Paratracheal NLNM', 'Con-Paratracheal NLNM', 'NLNM-prRLN']
data_enc[count_cols] = data_enc[count_cols].clip(lower=0).round().astype(int)

# 保存处理后的数据
data_enc.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/2.V1T编码后_插补.csv', index=False)


# %% [markdown]
# 传统的插补法

# %%
##复制右边的数据的路径用于数据的读取
data_enc=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/1.总T编码后.csv')

# %%
#2.2众数填补的缺失值
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# 创建并拟合填充器
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_encImpute = pd.DataFrame(imp.fit_transform(data_enc))

# 设置列名
data_encImpute.columns = data_enc.columns

# %%
#2.3整合
data_encImpute

# %%
#将插补后的数据保存下来
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/2.总编码后填补缺失值后.csv')

# %% [markdown]
# 3、数值数据矫正和归一化

# %%
#1.3数值数据校准和归一化
data_encImpute=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/0.数据/1.总Q编码后_插补.csv')

# %%
data_scale=data_encImpute

# %%
#3.1
target=data_encImpute['Pretracheal LNM'].astype(int)


# %%
##3.2
target.value_counts()

# %%
from sklearn import preprocessing

# %%
#第一种方法
scaler=preprocessing.StandardScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns

# %%
data_scaled


# %%
#第二种方法
scaler=preprocessing.RobustScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns

# %%
data_scaled

# %%
#第三种方法
scaler=preprocessing.MinMaxScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns

# %%
data_scaled

# %%
#将矫正后的数据保存下来
data_scaled.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# %% [markdown]
# 4、降维（减少因子之间的多重共线性的问题）

# %%
data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# %%
#查看特征变量
data.iloc[:,[0,2,4]]

# %%
data.shape

# %%
data.info()

# %% [markdown]
# #4.1移除低方差特征

# %%
data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')


#4.1移除方差特征
data_feature = data[['Age','Sex','BMI','Tumor border','Aspect ratio','Composition','Internal echo pattern','Internal echo homogeneous','Calcification',
                     'Tumor internal vascularization','Tumor Peripheral blood flow','Size','Location','Mulifocality','Hashimoto','Extrathyroidal extension',
                     'Side of position','Prelaryngeal LNM','Paratracheal LNM','Con-Paratracheal LNM','LNM-prRLN',
                     
                     'age','bmi','size','Prelaryngeal LNMR','Prelaryngeal NLNM','Paratracheal LNMR','Paratracheal NLNM','Con-Paratracheal LNMR',
                     'Con-Paratracheal NLNM','LNMR-prRLN','NLNM-prRLN',
    
    ]]


data_target=data['Pretracheal LNM']



# %%
from sklearn.feature_selection import VarianceThreshold


# 创建方差阈值选择器
sel = VarianceThreshold(threshold=(.9 * (1 - .9))) #如果觉得不够可以把特征筛选选大或者选小

# 应用方差阈值选择器到数据
data_sel = sel.fit_transform(data)


# %%
data_sel

# %%
a=sel.get_support(indices=True)

# %%

a

# %%

data.iloc[:,a]

# %%
data_sel=data.iloc[:,a]

# %%
data_sel.info()

# %% [markdown]
# #4.2单变量特征的选择

# %%
#单变量特征选择
from sklearn.feature_selection import SelectKBest, chi2



# %%

data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')


#4.1移除方差特征
data_feature = data[['Age','Sex','BMI','Tumor border','Aspect ratio','Composition','Internal echo pattern','Internal echo homogeneous','Calcification',
                     'Tumor internal vascularization','Tumor Peripheral blood flow','Size','Location','Mulifocality','Hashimoto','Extrathyroidal extension',
                     'Side of position','Prelaryngeal LNM','Paratracheal LNM','Con-Paratracheal LNM','LNM-prRLN',
                     
                     'age','bmi','size','Prelaryngeal LNMR','Prelaryngeal NLNM','Paratracheal LNMR','Paratracheal NLNM','Con-Paratracheal LNMR',
                     'Con-Paratracheal NLNM','LNMR-prRLN','NLNM-prRLN',
    
    ]]


data_target=data['Pretracheal LNM']
data_feature.shape
data_target.unique()#二分类

# %%
set_kit=SelectKBest(chi2,k=15)#选取k值最高的10(5)个元素
data_sel=set_kit.fit_transform(data_feature,data_target)
data_sel.shape

# %%
a=set_kit.get_support(indices=True)

# %%
a

# %%
data_sel=data_feature.iloc[:,a]

# %%
data_sel

# %%
data_sel.info()

# %% [markdown]
# #4.3递归特征消除RFE---线性模型。

# %%
#递归特征消除
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR #知识向量回归模型
from sklearn.model_selection import cross_val_score #知识向量回归模型


# %%
data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')


#4.1移除方差特征
data_feature = data[['Age','Sex','BMI','Tumor border','Aspect ratio','Composition','Internal echo pattern','Internal echo homogeneous','Calcification',
                     'Tumor internal vascularization','Tumor Peripheral blood flow','Size','Location','Mulifocality','Hashimoto','Extrathyroidal extension',
                     'Side of position','Prelaryngeal LNM','Paratracheal LNM','Con-Paratracheal LNM','LNM-prRLN',
                     
                     'age','bmi','size','Prelaryngeal LNMR','Prelaryngeal NLNM','Paratracheal LNMR','Paratracheal NLNM','Con-Paratracheal LNMR',
                     'Con-Paratracheal NLNM','LNMR-prRLN','NLNM-prRLN',
    
    ]]


data_target=data['Pretracheal LNM']
data_feature.shape
data_target.unique()#二分类

# %%
estimator=SVR(kernel='linear')
sel=RFE(estimator,n_features_to_select=14,step=1) #筛选出的变量，每递增一次就要删除一个特征，把权重最低的删掉

# %%
data_target=data['Pretracheal LNM']
data_target.unique()#二分类

# %%
sel.fit(data_feature,data_target)#跑得太久，1个小时以上，要跑过的导入进去

# %%
a=sel.get_support(indices=True)

# %%
a

# %%
data_sel=data_feature.iloc[:,a]

# %%
data_sel

# %%
data_sel.info()

# %% [markdown]
# #4.4RFECV-结合了交叉验证

# %%
#RFECV
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR #知识向量回归模型
from sklearn.model_selection import cross_val_score #知识向量回归模型

# %%

RFC_ = RandomForestClassifier()  # 随机森林
RFC_.fit(data_sel, data_target)  # 拟合模型
c = RFC_.feature_importances_  # 特征重要性
print('重要性：')
print(c)

# %%

selector = RFECV(RFC_, step=1, cv=10,min_features_to_select=10)  # 采用交叉验证cv就是10倍交叉验证，每次排除一个特征，筛选出最优特征
selector.fit(data_sel, data_target)
X_wrapper = selector.transform(data_sel)  # 最优特征
score = cross_val_score(RFC_, X_wrapper, data_target, cv=5).mean()  # 最优特征分类结果
print(score)
print('最佳数量和排序')
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)


# %%
print(selector.support_)
feature_names = data_sel.columns
selected_features = feature_names[selector.support_]
print(selected_features)
print(selector.ranking_)

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(selector.ranking_)), selector.ranking_)
plt.xticks(range(len(selector.ranking_)), feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Ranking')
plt.title('Feature Importance Ranking')
plt.show()

# %%
!pip install matplotlib
import matplotlib.pyplot as plt
#绘图：

score = []
best_score = 0
best_features = 0

for i in range(1, 8):
    X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(data_sel, data_target)  # 最优特征
    once = cross_val_score(RFC_, X_wrapper, data_target, cv=10).mean()
    score.append(once)
    
    if once > best_score:
        best_score = once
        best_features = i
    
    print("当前最高得分:", best_score)
    print("最佳特征数量:", best_features)
    print("得分列表:", score)
    
plt.figure(figsize=[20, 5])
plt.plot(range(1, 8), score)
plt.show()
from sklearn.model_selection import StratifiedKFold
rfecv=RFECV(estimator=RFC_,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(data,data_target)
print("最优特征数量：%d" % rfecv.n_features_)
print("选择的特征：", rfecv.support_)
print("特征排名：", rfecv.ranking_)
print("Optimal number of features: %d" % selector.n_features_)

# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
plt.show()
print("Optimal number of features: %d" % rfecv.n_features_)

# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()

# %%
rfecv.get_support(indices=True)

# %%
a

# %%
data.iloc[:,a]

# %%
data_sel=data.iloc[:,a]

# %%
data_sel.info()

# %% [markdown]
# #4.5基于L1范数的特征选取-seleformodel 模型特征选择-----分类的线性回归，SVM LINEARSVC

# %%
#L1-base
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression #知识向量回归模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold

# %%
clf = LogisticRegression()
clf.fit(data_feature, data_target)

model = SelectFromModel(clf, prefit=True)
data_new = model.transform(data_feature)

# %%
model.get_support(indices=True)

# %%
a=model.get_support(indices=True)

# %%
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns

# %%
data_featurenew=data_features.iloc[:,a]


# %%
data_featurenew

# %%
data_featurenew.info()

# %% [markdown]
# #4.6基于树模型--

# %%
#tree-base

data=pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')


#4.1移除方差特征
data_feature = data[['Age','Sex','BMI','Tumor border','Aspect ratio','Composition','Internal echo pattern','Internal echo homogeneous','Calcification',
                     'Tumor internal vascularization','Tumor Peripheral blood flow','Size','Location','Mulifocality','Hashimoto','Extrathyroidal extension',
                     'Side of position','Prelaryngeal LNM','Paratracheal LNM','Con-Paratracheal LNM','LNM-prRLN',
                     
                     'age','bmi','size','Prelaryngeal LNMR','Prelaryngeal NLNM','Paratracheal LNMR','Paratracheal NLNM','Con-Paratracheal LNMR',
                     'Con-Paratracheal NLNM','LNMR-prRLN','NLNM-prRLN',
    
    ]]


data_target=data['Pretracheal LNM']
data_feature.shape
data_target.unique()#二分类

# %%
clf = ExtraTreesClassifier()
clf.fit(data_feature, data_target)
clf.feature_importances_

# %%
model=SelectFromModel(clf,prefit=True)
x_new=model.transform(data_feature)

# %%
x_new


# %%
model.get_support(indices=True)

# %%
a=model.get_support(indices=True)

# %%
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]

# %%
data_featurenew

# %%
data_featurenew.info()

# %% [markdown]
# 筛选变量

# %%
pip install numpy pandas scikit-learn matplotlib xgboost lightgbm catboost shap seaborn


# %%
pip install rpart

# %%
####看这里筛选变量
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier
# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')
# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
    'Logistic Regression': (LogisticRegression(random_state=33), {'C': [0.01, 0.1, 1, 10, 100]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'max_depth': [1, 3, 4], 'min_samples_split': [1, 5,7]}),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 100, 200], 'max_depth': [1, 3, 5], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 定义颜色列表
colors = [
    '#FF6F6F',   
    '#FF9F9F',  
    '#FF4F4F',  
    '#FF1F1F',  
    '#FFBF80',  
    '#FF9F40',  
    '#FF7F00',  
    '#FF4F00',  
    '#FFFF80',  
    '#80FF80',  
    '#80B3FF',  
    '#B380FF',   
    '#D3D3D3',  
]


# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None

# 创建评价指标的空列表
train_accuracy_scores = []
train_auc_scores = []
train_precision_scores = []
train_specificity_scores = []
train_sensitivity_scores = []
train_npv_scores = []
train_ppv_scores = []
train_recall_scores = []
train_f1_scores = []
train_fpr_scores = []
train_rmse_scores = []
train_r2_scores = []
train_mae_scores = []
train_tn_scores = []
train_fp_scores = []
train_fn_scores = []
train_tp_scores = []
train_lift_scores = []
train_brier_scores = []
train_kappa_scores = []
# 拟合模型并绘制训练集的ROC曲线
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_train_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_train_pred_prob = best_model_temp.decision_function(class_x_tra)
    
    
    
    
    
    # 计算AUC值
    auc = roc_auc_score(class_y_tra, y_train_pred_prob)

    # 如果当前模型的AUC值是最高的，则更新最佳模型和最佳AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_model_temp
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(class_y_tra, y_train_pred_prob)
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

     #计算其他评价指标
    train_y_pred = best_model_temp.predict(class_x_tra)
    train_accuracy = accuracy_score(class_y_tra, train_y_pred)
    train_precision = precision_score(class_y_tra, train_y_pred)
    train_cm = confusion_matrix(class_y_tra, train_y_pred)
    train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
    train_specificity = train_tn / (train_tn + train_fp)
    train_sensitivity = recall_score(class_y_tra, train_y_pred)
    train_npv = train_tn / (train_tn + train_fn)
    train_ppv = train_tp / (train_tp + train_fp)
    train_recall = train_sensitivity
    train_f1 = f1_score(class_y_tra, train_y_pred)
    train_fpr = train_fp / (train_fp + train_tn)
    train_rmse = mean_squared_error(class_y_tra, y_train_pred_prob, squared=False)
    train_r2 = r2_score(class_y_tra, y_train_pred_prob)
    train_mae = mean_absolute_error(class_y_tra, y_train_pred_prob)
    train_auc = roc_auc_score(class_y_tra, y_train_pred_prob)
    train_lift = average_precision_score(class_y_tra, y_train_pred_prob) / (sum(class_y_tra) / len(class_y_tra))
    train_kappa = cohen_kappa_score(class_y_tra, train_y_pred)
    train_brier = brier_score_loss(class_y_tra, y_train_pred_prob)
    
     
    # 将评价指标添加到列表中
    train_accuracy_scores.append(train_accuracy)
    train_auc_scores.append(train_auc)
    train_precision_scores.append(train_precision)
    train_specificity_scores.append(train_specificity)
    train_sensitivity_scores.append(train_sensitivity)
    train_npv_scores.append(train_npv)
    train_ppv_scores.append(train_ppv)
    train_recall_scores.append(train_recall)
    train_f1_scores.append(train_f1)
    train_fpr_scores.append(train_fpr)
    train_rmse_scores.append(train_rmse)
    train_r2_scores.append(train_r2)
    train_mae_scores.append(train_mae)
    train_tn_scores.append(train_tn)
    train_fp_scores.append(train_fp)
    train_fn_scores.append(train_fn)
    train_tp_scores.append(train_tp)
    train_lift_scores.append(train_lift)
    train_brier_scores.append(train_brier)
    train_kappa_scores.append(train_kappa)    
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC For Prelaryngeal Lymph Node Metastasis-Tree-based Feature Selection(Train set)')
plt.legend(loc='lower right')
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/图3/Figure2-F1-ROC All Machine Learning Algorithms-Tree-based Feature Selection{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()
# 打印最佳模型的名称和AUC值
print(f"最佳模型: {best_model_name} with AUC = {best_auc}")
# 使用最佳模型在验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_test_pred_prob = best_model.predict_proba(class_x_test)[:, 1]
else:
    y_test_pred_prob = best_model.decision_function(class_x_test)
# 计算验证集上的AUC值
test_auc = roc_auc_score(class_y_test, y_test_pred_prob)
# 打印验证集上的AUC值
print(f"测试集上的AUC = {test_auc}")
# 使用最佳模型在外验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_val_pred_prob1 = best_model.predict_proba(val_features1)[:, 1]
else:
    y_val_pred_prob1 = best_model.decision_function(val_features1)

# 计算外验证集上的AUC值
val_auc1 = roc_auc_score(val_target1, y_val_pred_prob1)
# 打印外验证集上的AUC值
print(f"外验证集上的AUC = {val_auc1}")
# 使用最佳模型在外验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_val_pred_prob2 = best_model.predict_proba(val_features2)[:, 1]
else:
    y_val_pred_prob2 = best_model.decision_function(val_features2)
# 计算外验证集上的AUC值
val_auc2 = roc_auc_score(val_target2, y_val_pred_prob2)
# 打印外验证集上的AUC值
print(f"外验证集上的AUC = {val_auc2}")
# 绘制验证集和外验证集的ROC曲线
fpr_test, tpr_test, _ = roc_curve(class_y_test, y_test_pred_prob)
fpr_val1, tpr_val1, _ = roc_curve(val_target1, y_val_pred_prob1)
fpr_val2, tpr_val2, _ = roc_curve(val_target2, y_val_pred_prob2)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#C9A51A', label='Train Set (AUC = %0.3f)' % best_auc)
plt.plot(fpr_test, tpr_test, color='#ECAC27', label='Test Set (AUC = %0.3f)' % test_auc)
plt.plot(fpr_val1, tpr_val1, color='#EDDE23', label='Validation Set1 (AUC = %0.3f)' % val_auc1)
plt.plot(fpr_val2, tpr_val2, color='#FFFF66', label='Validation Set2 (AUC = %0.3f)' % val_auc2)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Prelaryngeal Lymph Node Metastasis-Tree-based Feature Selection')
plt.legend(loc='lower right')
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/图3/Figure2-F2-ROC All Machine Learning Algorithms-Tree-based Feature Selection{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()
# 净收益计算函数
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        if thresh == 1.0:  # 避免除以0
            net_benefit = 0
        else:
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)
    return net_benefits
# 决策阈值
decision_thresholds = np.linspace(0, 1, 101)
# 计算净收益
net_benefits_train = calculate_net_benefit(class_y_tra, y_train_pred_prob, decision_thresholds)
net_benefits_test = calculate_net_benefit(class_y_test, y_test_pred_prob, decision_thresholds)
net_benefits_val1 = calculate_net_benefit(val_target1, y_val_pred_prob1, decision_thresholds)
net_benefits_val2 = calculate_net_benefit(val_target2, y_val_pred_prob2, decision_thresholds)
# 计算所有人都进行干预时的净收益
all_positive_train = np.ones_like(class_y_tra)
all_positive_test = np.ones_like(class_y_test)
all_positive_val1 = np.ones_like(val_target1)
all_positive_val2 = np.ones_like(val_target2)
net_benefit_all_train = calculate_net_benefit(class_y_tra, all_positive_train, decision_thresholds)
net_benefit_all_test = calculate_net_benefit(class_y_test, all_positive_test, decision_thresholds)
net_benefit_all_val1 = calculate_net_benefit(val_target1, all_positive_val1, decision_thresholds)
net_benefit_all_val2 = calculate_net_benefit(val_target2, all_positive_val2, decision_thresholds)
# 绘制DCA曲线
plt.figure(figsize=(8, 6))
plt.plot(decision_thresholds, net_benefits_train, color='#C9A51A', lw=2, label='Training set')
plt.plot(decision_thresholds, net_benefits_test, color='#ECAC27', lw=2, label='Test set')
plt.plot(decision_thresholds, net_benefits_val1, color='#EDDE23', lw=2, label='Validation set1')
plt.plot(decision_thresholds, net_benefits_val2, color='#FFFF66', lw=2, label='Validation set2')
plt.plot(decision_thresholds, net_benefit_all_val1, color='gray', lw=2, linestyle='--', label='All')
plt.plot(decision_thresholds, np.zeros_like(decision_thresholds), color='black', lw=2, linestyle='-', label='None')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title('DCA for Prelaryngeal Lymph Node Metastasis-Tree-based Feature Selection', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.xlim(0, 0.8)
plt.ylim(-0.1, 0.2)
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [ 300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/图3/Figure2-F3-dca All Machine Learning Algorithms-Tree-based Feature Selection{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()
# 打印XGBoost最佳参数
print("最佳模型的参数设置:")
print(grid_search.best_params_)
# 创建训练集评价指标的DataFrame
train_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': train_accuracy_scores,
    'AUC': train_auc_scores,
    'Precision': train_precision_scores,
    'Specificity': train_specificity_scores,
    'Sensitivity': train_sensitivity_scores,
    'Negative Predictive Value': train_npv_scores,
    'Positive Predictive Value': train_ppv_scores,
    'Recall': train_recall_scores,
    'F1 Score': train_f1_scores,
    'False Positive Rate': train_fpr_scores,
    'RMSE': train_rmse_scores,
    'R2': train_r2_scores,
    'MAE': train_mae_scores,
    'True Negatives': train_tn_scores,
    'False Positives': train_fp_scores,
    'False Negatives': train_fn_scores,
    'True Positives': train_tp_scores,
    'Lift': train_lift_scores,
    'Brier Score': train_brier_scores,
    'Kappa': train_kappa_scores,
})
 #显示训练集评价指标DataFrame
print(train_metrics_df)

 #将训练集评价指标DataFrame导出为CSV文件
train_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/1.F气管前Tree-based Feature Selection.csv', index=False)


# %% [markdown]
# ##1.1.1训练集的ROC曲线

# %% [markdown]
# 最终选择的这一个版本

# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier

# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')
# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
    'Logistic Regression': (LogisticRegression(random_state=33), {'C': [0.01, 0.1, 1, 10, 100]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'max_depth': [1, 3, 4], 'min_samples_split': [1, 5,7]}),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 100, 200], 'max_depth': [1, 3, 5], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 定义颜色列表
colors = [
    '#F44336',  # Very Light Red
    '#FB8C00',  # Light Red
    '#FDD835',  # Light Pink Red
    '#43A047',  # Soft Red
    '#1E88E5',  # Warm Light Red
    '#8E24AA',  # Bright Red
    '#F06292',  # Strong Red
    '#FBC02D',
    '#FFAB91',
    '#00ACC1',  # Pure Red
    '#D81B60',  # Dark Red
    '#00796B',  # Darker Red
    '#6D4C41',  # Deep Red
]



# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None

# 创建评价指标的空列表
train_accuracy_scores = []
train_auc_scores = []
train_precision_scores = []
train_specificity_scores = []
train_sensitivity_scores = []
train_npv_scores = []
train_ppv_scores = []
train_recall_scores = []
train_f1_scores = []
train_fpr_scores = []
train_rmse_scores = []
train_r2_scores = []
train_mae_scores = []
train_tn_scores = []
train_fp_scores = []
train_fn_scores = []
train_tp_scores = []
train_lift_scores = []
train_brier_scores = []
train_kappa_scores = []
# 拟合模型并绘制训练集的ROC曲线
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_train_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_train_pred_prob = best_model_temp.decision_function(class_x_tra)

    # 计算AUC值
    auc = roc_auc_score(class_y_tra, y_train_pred_prob)

    # 如果当前模型的AUC值是最高的，则更新最佳模型和最佳AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_model_temp

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(class_y_tra, y_train_pred_prob)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

     #计算其他评价指标
    train_y_pred = best_model_temp.predict(class_x_tra)
    train_accuracy = accuracy_score(class_y_tra, train_y_pred)
    train_precision = precision_score(class_y_tra, train_y_pred)
    train_cm = confusion_matrix(class_y_tra, train_y_pred)
    train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
    train_specificity = train_tn / (train_tn + train_fp)
    train_sensitivity = recall_score(class_y_tra, train_y_pred)
    train_npv = train_tn / (train_tn + train_fn)
    train_ppv = train_tp / (train_tp + train_fp)
    train_recall = train_sensitivity
    train_f1 = f1_score(class_y_tra, train_y_pred)
    train_fpr = train_fp / (train_fp + train_tn)
    train_rmse = mean_squared_error(class_y_tra, y_train_pred_prob, squared=False)
    train_r2 = r2_score(class_y_tra, y_train_pred_prob)
    train_mae = mean_absolute_error(class_y_tra, y_train_pred_prob)
    train_auc = roc_auc_score(class_y_tra, y_train_pred_prob)
    train_lift = average_precision_score(class_y_tra, y_train_pred_prob) / (sum(class_y_tra) / len(class_y_tra))
    train_kappa = cohen_kappa_score(class_y_tra, train_y_pred)
    train_brier = brier_score_loss(class_y_tra, y_train_pred_prob)
    
    
    # 将评价指标添加到列表中
    train_accuracy_scores.append(train_accuracy)
    train_auc_scores.append(train_auc)
    train_precision_scores.append(train_precision)
    train_specificity_scores.append(train_specificity)
    train_sensitivity_scores.append(train_sensitivity)
    train_npv_scores.append(train_npv)
    train_ppv_scores.append(train_ppv)
    train_recall_scores.append(train_recall)
    train_f1_scores.append(train_f1)
    train_fpr_scores.append(train_fpr)
    train_rmse_scores.append(train_rmse)
    train_r2_scores.append(train_r2)
    train_mae_scores.append(train_mae)
    train_tn_scores.append(train_tn)
    train_fp_scores.append(train_fp)
    train_fn_scores.append(train_fn)
    train_tp_scores.append(train_tp)
    train_lift_scores.append(train_lift)
    train_brier_scores.append(train_brier)
    train_kappa_scores.append(train_kappa)
    
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC For Pretracheal Lymph Node Metastasis (Train set)')
plt.legend(loc='lower right')
# 保存图像为TIFF格式

formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure1-C1-Pretracheal-roc-Train_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# 打印最佳模型的名称和AUC值
print(f"最佳模型: {best_model_name} with AUC = {best_auc}")
# 打印最佳模型的名称和参数
print(f"最佳模型: {best_model_name}")
print("最佳模型参数设置:")
print(best_model.get_params())

# 使用最佳模型在验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_test_pred_prob = best_model.predict_proba(class_x_test)[:, 1]
else:
    y_test_pred_prob = best_model.decision_function(class_x_test)
# 计算验证集上的AUC值
test_auc = roc_auc_score(class_y_test, y_test_pred_prob)
# 打印验证集上的AUC值
print(f"测试集上的AUC = {test_auc}")
# 使用最佳模型在外验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_val_pred_prob1 = best_model.predict_proba(val_features1)[:, 1]
else:
    y_val_pred_prob1 = best_model.decision_function(val_features1)

# 计算外验证集上的AUC值
val_auc1 = roc_auc_score(val_target1, y_val_pred_prob1)
# 打印外验证集上的AUC值
print(f"外验证集上的AUC = {val_auc1}")
# 使用最佳模型在外验证集上进行评估
if hasattr(best_model, 'predict_proba'):
    y_val_pred_prob2 = best_model.predict_proba(val_features2)[:, 1]
else:
    y_val_pred_prob2 = best_model.decision_function(val_features2)
# 计算外验证集上的AUC值
val_auc2 = roc_auc_score(val_target2, y_val_pred_prob2)
# 打印外验证集上的AUC值
print(f"外验证集上的AUC = {val_auc2}")
# 绘制验证集和外验证集的ROC曲线
fpr_test, tpr_test, _ = roc_curve(class_y_test, y_test_pred_prob)
fpr_val1, tpr_val1, _ = roc_curve(val_target1, y_val_pred_prob1)
fpr_val2, tpr_val2, _ = roc_curve(val_target2, y_val_pred_prob2)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#C9A51A', label='Train Set (AUC = %0.3f)' % best_auc)
plt.plot(fpr_test, tpr_test, color='#ECAC27', label='Test Set (AUC = %0.3f)' % test_auc)
plt.plot(fpr_val1, tpr_val1, color='#EDDE23', label='Validation Set1 (AUC = %0.3f)' % val_auc1)
plt.plot(fpr_val2, tpr_val2, color='#FFFF66', label='Validation Set2 (AUC = %0.3f)' % val_auc2)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Pretracheal Lymph Node Metastasis-XGBOOST')
plt.legend(loc='lower right')
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure6-C1-Pretracheal-roc-ALL-SET_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# 净收益计算函数
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        if thresh == 1.0:  # 避免除以0
            net_benefit = 0
        else:
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)
    return net_benefits
# 决策阈值
decision_thresholds = np.linspace(0, 1, 101)
# 计算净收益
net_benefits_train = calculate_net_benefit(class_y_tra, y_train_pred_prob, decision_thresholds)
net_benefits_test = calculate_net_benefit(class_y_test, y_test_pred_prob, decision_thresholds)
net_benefits_val1 = calculate_net_benefit(val_target1, y_val_pred_prob1, decision_thresholds)
net_benefits_val2 = calculate_net_benefit(val_target2, y_val_pred_prob2, decision_thresholds)
# 计算所有人都进行干预时的净收益
all_positive_train = np.ones_like(class_y_tra)
all_positive_test = np.ones_like(class_y_test)
all_positive_val1 = np.ones_like(val_target1)
all_positive_val2 = np.ones_like(val_target2)
net_benefit_all_train = calculate_net_benefit(class_y_tra, all_positive_train, decision_thresholds)
net_benefit_all_test = calculate_net_benefit(class_y_test, all_positive_test, decision_thresholds)
net_benefit_all_val1 = calculate_net_benefit(val_target1, all_positive_val1, decision_thresholds)
net_benefit_all_val2 = calculate_net_benefit(val_target2, all_positive_val2, decision_thresholds)
# 绘制DCA曲线
plt.figure(figsize=(8, 6))
plt.plot(decision_thresholds, net_benefits_train, color='#C9A51A', lw=2, label='Training set')
plt.plot(decision_thresholds, net_benefits_test, color='#ECAC27', lw=2, label='Test set')
plt.plot(decision_thresholds, net_benefits_val1, color='#EDDE23', lw=2, label='Validation set1')
plt.plot(decision_thresholds, net_benefits_val2, color='#FFFF66', lw=2, label='Validation set2')
plt.plot(decision_thresholds, net_benefit_all_val1, color='gray', lw=2, linestyle='--', label='All')
plt.plot(decision_thresholds, np.zeros_like(decision_thresholds), color='black', lw=2, linestyle='-', label='None')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title('DCA for Pretracheal Lymph Node Metastasis-XGBOOST', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.xlim(0, 0.8)
plt.ylim(-0.1, 0.4)
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [ 300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure6-C2-Pretracheal-DCA-ALL-SET_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()



# 创建训练集评价指标的DataFrame
train_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': train_accuracy_scores,
    'AUC': train_auc_scores,
    'Precision': train_precision_scores,
    'Specificity': train_specificity_scores,
    'Sensitivity': train_sensitivity_scores,
    'Negative Predictive Value': train_npv_scores,
    'Positive Predictive Value': train_ppv_scores,
    'Recall': train_recall_scores,
    'F1 Score': train_f1_scores,
    'False Positive Rate': train_fpr_scores,
    'RMSE': train_rmse_scores,
    'R2': train_r2_scores,
    'MAE': train_mae_scores,
    'True Negatives': train_tn_scores,
    'False Positives': train_fp_scores,
    'False Negatives': train_fn_scores,
    'True Positives': train_tp_scores,
    'Lift': train_lift_scores,
    'Brier Score': train_brier_scores,
    'Kappa': train_kappa_scores,
})

# 显示训练集评价指标DataFrame
print(train_metrics_df)

# 将训练集评价指标DataFrame导出为CSV文件

# 将训练集评价指标DataFrame导出为CSV文件
train_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/1.C.Pretracheal训练集的评价指标.csv', index=False)



# %% [markdown]
# ##1.1.3训练集的DCA曲线

# %%
#训练集的决策曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
tra_net_benefit = []

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_tra_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_tra_pred_prob = best_model_temp.decision_function(class_x_tra)

    tra_model_net_benefit = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        tra_predictions = (y_tra_pred_prob >= threshold).astype(int)
        tp = np.sum((class_y_tra == 1) & (tra_predictions == 1))
        fp = np.sum((class_y_tra == 0) & (tra_predictions == 1))
        fn = np.sum((class_y_tra == 1) & (tra_predictions == 0))
        tn = np.sum((class_y_tra == 0) & (tra_predictions == 0))
        
        net_benefit = (tp / len(class_y_tra)) - (fp / len(class_y_tra)) * (threshold / (1 - threshold))
        tra_model_net_benefit.append(net_benefit)
        
    tra_net_benefit.append(tra_model_net_benefit)

# 转换为数组
tra_net_benefit = np.array(tra_net_benefit)

# 计算所有人都进行干预时的净收益
tra_all_predictions = np.ones_like(class_y_tra)  # 将所有预测标记为阳性（正类）
tp_all = np.sum((class_y_tra == 1) & (tra_all_predictions == 1))
fp_all = np.sum((class_y_tra == 0) & (tra_all_predictions == 1))

net_benefit_all = (tp_all / len(class_y_tra)) - (fp_all / len(class_y_tra)) * (thresholds / (1 - thresholds))
net_benefit_none = np.zeros_like(thresholds)
names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Extra Trees',
    'AdaBoost',
    'Gradient Boosting',
    'HistGradientBoosting',
    'XGBoost', 
    'CatBoost',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Neural Network',
    'Gaussian Naive Bayes',
]

# 绘制DCA曲线
for i in range(tra_net_benefit.shape[0]):
    plt.plot(thresholds, tra_net_benefit[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, net_benefit_none, color='black', linestyle='-', label='None')
plt.plot(thresholds, net_benefit_all, color='gray', linestyle='--', label='All')


# 设置y轴的限制
plt.xlim(0, 0.8)
plt.ylim(-0.1,0.20)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA For Pretracheal Lymph Node Metastasis(Train set)')
plt.legend(loc='upper right')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)

# 显示图形
# 保存图像为TIFF格式

formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure2-C1-Pretracheal-DCA-Train_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %% [markdown]
# ##1.1.4训练集的校准曲线

# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# 创建一个空列表来存储每个模型的校准曲线和 Brier Score
train_calibration_curves = []
train_brier_scores = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_train_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_train_pred_prob = best_model_temp.decision_function(class_x_tra)

    # 计算校准曲线
    train_fraction_of_positives, train_mean_predicted_value = calibration_curve(class_y_tra, y_train_pred_prob, n_bins=10)
    train_calibration_curves.append((train_fraction_of_positives, train_mean_predicted_value, name, color))

    # 计算 Brier 分数
    train_brier_score = brier_score_loss(class_y_tra, y_train_pred_prob)
    train_brier_scores.append((name, train_brier_score))

    # 打印 Brier 分数
    print(f'{name} - Train Brier Score: {train_brier_score:.3f}')

# 绘制校准曲线和 Brier Score
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in train_calibration_curves:
    train_fraction_of_positives, train_mean_predicted_value, name, color = curve
    
    # 获取对应模型的 Brier Score
    train_brier_score = next((score for model, score in train_brier_scores if model == name), None)
    
    # 将 Brier Score 赋予线颜色标注名称的后面
    if train_brier_score is not None:
        name += f' (Train Brier Score: {train_brier_score:.3f})'
    
    ax1.plot(train_mean_predicted_value, train_fraction_of_positives, "s-", label=name, color=color)
    
# 绘制 "Perfectly calibrated" 曲线
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Pretracheal Lymph Node Metastasis(Train set)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure3-C1-Pretracheal-CC-Train_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()


# %% [markdown]
# ##1.1.5训练集的精确召回曲线

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# 初始化存储精确召回曲线和平均精确度的列表
train_precision_recall_curves = []
train_average_precision_scores = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_train_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_train_pred_prob = best_model_temp.decision_function(class_x_tra)

    # 计算精确召回曲线
    train_precision, train_recall, _ = precision_recall_curve(class_y_tra, y_train_pred_prob)
    train_average_precision = average_precision_score(class_y_tra, y_train_pred_prob)

    # 存储结果
    train_precision_recall_curves.append((train_precision, train_recall, f'{name} (AUPR: {train_average_precision:.3f})', color))
    train_average_precision_scores.append((f'{name} (AUPR: {train_average_precision:.3f})', train_average_precision))

    # 打印平均精确度
    print(f'{name} - Train Average Precision: {train_average_precision:.3f}')

# 绘制精确召回曲线
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in train_precision_recall_curves:
    train_precision, train_recall, name, color = curve
    ax2.plot(train_recall, train_precision, "-", color=color, label=name)

# 添加随机猜测曲线
plt.plot([0, 1], [class_y_tra.mean(), class_y_tra.mean()], linestyle='--', color='black', label='Random Guessing')

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend(loc="lower left")
ax2.grid(True)

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Precision Recall Curves For Pretracheal Lymph Node Metastasis(Train set)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure4-C1-Pretracheal-PRC-Train_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()


# %% [markdown]
# 学习曲线

# %% [markdown]
# ##2.1.1验证集-内验证集的ROC曲线

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier

# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')
# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
    'Logistic Regression':  (DecisionTreeClassifier(random_state=33), {'max_depth': [1, 3, 5, 7], 'min_samples_split': [1,2,3],}),
    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (AdaBoostClassifier(random_state=33), {'n_estimators': [30, 50, 70], 'learning_rate': [0.01, 0.3, 0.7]}),
    'Extra Trees': (AdaBoostClassifier(random_state=33), {'n_estimators': [10, 60, 80], 'learning_rate': [0.03, 0.5, 0.8]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [1, 3, 10], 'learning_rate': [0.04, 0.05],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [1, 3, 5], }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1,10], }),
    'K-Nearest Neighbors': (AdaBoostClassifier(random_state=33), {'n_estimators': [30, 40, 50], 'learning_rate': [0.01, 0.03, 0.06]}),
    'Neural Network': (AdaBoostClassifier(random_state=33), {'n_estimators': [10, 20, 30], 'learning_rate': [0.01, 0.03, 0.05]}),
    'Gaussian Naive Bayes': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 50, 60], 'learning_rate': [0.02, 0.07, 0.09]}),
}

# 定义颜色列表
colors = [
    '#F44336',  # Very Light Red
    '#FB8C00',  # Light Red
    '#FDD835',  # Light Pink Red
    '#43A047',  # Soft Red
    '#1E88E5',  # Warm Light Red
    '#8E24AA',  # Bright Red
    '#F06292',  # Strong Red
    '#FBC02D',
    '#FFAB91',
    '#00ACC1',  # Pure Red
    '#D81B60',  # Dark Red
    '#00796B',  # Darker Red
    '#6D4C41',  # Deep Red
]




# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None
# 创建评价指标的空列表
Test_accuracy_scores = []
Test_auc_scores = []
Test_precision_scores = []
Test_specificity_scores = []
Test_sensitivity_scores = []
Test_npv_scores = []
Test_ppv_scores = []
Test_recall_scores = []
Test_f1_scores = []
Test_fpr_scores = []
Test_rmse_scores = []
Test_r2_scores = []
Test_mae_scores = []
Test_tn_scores = []
Test_fp_scores = []
Test_fn_scores = []
Test_tp_scores = []
Test_lift_scores = []
Test_brier_scores = []
Test_kappa_scores = []

# 拟合模型并绘制ROC曲线
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_test_pred_prob = best_model_temp.predict_proba(class_x_test)[:, 1]
    else:
        y_test_pred_prob = best_model_temp.decision_function(class_x_test)

    # 计算AUC值
    auc = roc_auc_score(class_y_test, y_test_pred_prob)

    # 如果当前模型的AUC值是最高的，则更新最佳模型和最佳AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_model_temp

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(class_y_test, y_test_pred_prob)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # 计算其他评价指标
    test_y_pred = best_model_temp.predict(class_x_test)
    test_accuracy = accuracy_score(class_y_test, test_y_pred)
    test_precision = precision_score(class_y_test, test_y_pred)
    test_cm = confusion_matrix(class_y_test, test_y_pred)
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
    test_specificity = test_tn / (test_tn + test_fp)
    test_sensitivity = recall_score(class_y_test, test_y_pred)
    test_npv = test_tn / (test_tn + test_fn)
    test_ppv = test_tp / (test_tp + test_fp)
    test_recall = test_sensitivity
    test_f1 = f1_score(class_y_test, test_y_pred)
    test_fpr = test_fp / (test_fp + test_tn)
    test_rmse = mean_squared_error(class_y_test, y_test_pred_prob, squared=False)
    test_r2 = r2_score(class_y_test, y_test_pred_prob)
    test_mae = mean_absolute_error(class_y_test, y_test_pred_prob)
    test_kappa = cohen_kappa_score(class_y_test, test_y_pred)
    test_auc = roc_auc_score(class_y_test, y_test_pred_prob)
    test_lift = average_precision_score(class_y_test, y_test_pred_prob) / (sum(class_y_test) / len(class_y_test))
    test_brier = brier_score_loss(class_y_test, y_test_pred_prob)

    # 将评价指标添加到列表中
    Test_accuracy_scores.append(test_accuracy)
    Test_auc_scores.append(auc)
    Test_precision_scores.append(test_precision)
    Test_specificity_scores.append(test_specificity)
    Test_sensitivity_scores.append(test_sensitivity)
    Test_npv_scores.append(test_npv)
    Test_ppv_scores.append(test_ppv)
    Test_recall_scores.append(test_recall)
    Test_f1_scores.append(test_f1)
    Test_fpr_scores.append(test_fpr)
    Test_rmse_scores.append(test_rmse)
    Test_r2_scores.append(test_r2)
    Test_mae_scores.append(test_mae)
    Test_tn_scores.append(test_tn)
    Test_fp_scores.append(test_fp)
    Test_fn_scores.append(test_fn)
    Test_tp_scores.append(test_tp)
    Test_lift_scores.append(test_lift)
    Test_brier_scores.append(test_brier)
    Test_kappa_scores.append(test_kappa)

plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC For Pretracheal Lymph Node Metastasis(Test set)')
plt.legend(loc='lower right')
# 保存图像为TIFF格式
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure1-C2-Pretracheal-roc-Test_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# 打印最佳模型的名称和AUC值
print(f"最佳模型: {best_model_name} with AUC = {best_auc}")

# 创建训练集评价指标的DataFrame
Val_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': Test_accuracy_scores,
    'AUC': Test_auc_scores,
    'Precision': Test_precision_scores,
    'Specificity': Test_specificity_scores,
    'Sensitivity': Test_sensitivity_scores,
    'Negative Predictive Value': Test_npv_scores,
    'Positive Predictive Value': Test_ppv_scores,
    'Recall': Test_recall_scores,
    'F1 Score': Test_f1_scores,
    'False Positive Rate': Test_fpr_scores,
    'RMSE': Test_rmse_scores,
    'R2': Test_r2_scores,
    'MAE': Test_mae_scores,
    'True Negatives': Test_tn_scores,
    'False Positives': Test_fp_scores,
    'False Negatives': Test_fn_scores,
    'True Positives':Test_tp_scores,
    'Lift': Test_lift_scores,
    'Brier Score': Test_brier_scores,
    'Kappa': Test_kappa_scores,  
})

# 显示训练集评价指标DataFrame
print(Val_metrics_df)

# 将训练集评价指标DataFrame导出为CSV文件
Val_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/1.2C.Pretracheal测试集(内验证)的评价指标新.csv', index=False)


# %% [markdown]
# ##2.1.3验证集-内验证集的决策曲线

# %%
#内验证集的决策曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
test_net_benefit = []


for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_test_pred_prob = best_model_temp.predict_proba(class_x_test)[:, 1]
    else:
        y_test_pred_prob = best_model_temp.decision_function(class_x_test)

    test_model_net_benefit = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        test_predictions = (y_test_pred_prob >= threshold).astype(int)
        tp = np.sum((class_y_test == 1) & (test_predictions == 1))
        fp = np.sum((class_y_test == 0) & (test_predictions == 1))
        fn = np.sum((class_y_test == 1) & (test_predictions == 0))
        tn = np.sum((class_y_test == 0) & (test_predictions == 0))
        
        net_benefit = (tp / len(class_y_test)) - (fp / len(class_y_test)) * (threshold / (1 - threshold))
        test_model_net_benefit.append(net_benefit)

    test_net_benefit.append(test_model_net_benefit)

# 转换为数组
test_net_benefit = np.array(test_net_benefit)

# 计算所有人都进行干预时的净收益
test_all_predictions = np.ones_like(class_y_test)  # 将所有预测标记为阳性（正类）
tp_all = np.sum((class_y_test == 1) & (test_all_predictions == 1))
fp_all = np.sum((class_y_test == 0) & (test_all_predictions == 1))

net_benefit_all = (tp_all / len(class_y_test)) - (fp_all / len(class_y_test)) * (thresholds / (1 - thresholds))
net_benefit_none = np.zeros_like(thresholds)

names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Extra Trees',
    'AdaBoost',
    'Gradient Boosting',
    'HistGradientBoosting',
    'XGBoost', 
    'CatBoost',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Neural Network',
    'Gaussian Naive Bayes',
]

# 绘制DCA曲线
for i in range(test_net_benefit.shape[0]):
    plt.plot(thresholds, test_net_benefit[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, net_benefit_none, color='black', linestyle='-', label='None')
plt.plot(thresholds, net_benefit_all, color='gray', linestyle='--', label='All')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
# 设置y轴的限制
plt.xlim(0, 0.8)
plt.ylim(-0.1,0.25)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA For Pretracheal Lymph Node Metastasis(Test set)')
plt.legend(loc='upper right')

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure2-C2-Pretracheal-DCA-Test_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %% [markdown]
# ##2.1.4验证集-内验证集的校准曲线

# %%
#内验证集的校准曲线
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import ttest_ind

# 创建一个空列表来存储每个模型的校准曲线和Brier Score
test_calibration_curves = []
test_brier_scores = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_test_pred_prob = best_model_temp.predict_proba(class_x_test)[:, 1]
    else:
        y_test_pred_prob = best_model_temp.decision_function(class_x_test)

    # 计算校准曲线
    test_fraction_of_positives, test_mean_predicted_value = calibration_curve(class_y_test, y_test_pred_prob, n_bins=10)
    test_calibration_curves.append((test_fraction_of_positives, test_mean_predicted_value, name, color))

    # 计算Brier分数
    test_brier_score = brier_score_loss(class_y_test, y_test_pred_prob)
    test_brier_scores.append((name, test_brier_score))

    # 打印Brier分数
    print(f'{name} - Brier Score: {test_brier_score:.3f}')

# 绘制校准曲线和Brier Score
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in test_calibration_curves:
    test_fraction_of_positives, test_mean_predicted_value, name, color = curve
    
    # 获取对应模型的Brier Score
    test_brier_score = next((score for model, score in test_brier_scores if model == name), None)
    
    # 将Brier Score赋予线颜色标注名称的后面
    if test_brier_score is not None:
        name += f' (Brier Score: {test_brier_score:.3f})'
    
    ax1.plot(test_mean_predicted_value, test_fraction_of_positives, "s-", label=name, color=color)
    
# 绘制"Perfectly calibrated"曲线
ax1.plot([0, 1], [0, 1], "k:",label="Perfectly calibrated")
    
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Pretracheal Lymph Node Metastasis(Test set)")
plt.tight_layout()
# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure3-C2-Pretracheal-CC-Test_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %% [markdown]
# ##2.1.5验证集-内验证集的绘制精确-召回曲线及AUPR值

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 初始化存储精确召回曲线和平均精确度的列表
test_precision_recall_curves = []
test_average_precision_scores = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算测试集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_test_pred_prob = best_model_temp.predict_proba(class_x_test)[:, 1]
    else:
        y_test_pred_prob = best_model_temp.decision_function(class_x_test)

    # 计算精确召回曲线
    test_precision, test_recall, _ = precision_recall_curve(class_y_test, y_test_pred_prob)
    test_average_precision = average_precision_score(class_y_test, y_test_pred_prob)

    # 存储结果
    test_precision_recall_curves.append((test_precision, test_recall, f'{name} (AUPR: {test_average_precision:.3f})', color))
    test_average_precision_scores.append((f'{name} (AUPR: {test_average_precision:.3f})', test_average_precision))

    # 打印平均精确度
    print(f'{name} - Average Precision: {test_average_precision:.3f}')

# 绘制精确召回曲线
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in test_precision_recall_curves:
    test_precision, test_recall, name, color = curve
    ax2.plot(test_recall, test_precision, "-", color=color, label=name)

# 添加随机猜测曲线
ax2.plot([0, 1], [class_y_test.mean(), class_y_test.mean()], linestyle='--', color='black', label='Random Guessing')

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend(loc="lower left")
ax2.grid(True)

plt.title("Precision Recall Curves For Pretracheal Lymph Node Metastasis(Test set)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure4-C2-Pretracheal-PRC-Train_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier

# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = [ 'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
   'Logistic Regression': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [1, 3,5], 
                                                            'min_samples_split': [1, 3, 5], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest':(ExtraTreesClassifier(random_state=33), {'n_estimators': [30, 70, 150], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [1, 3, 5], }),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 100, 200], 'max_depth': [1, 3, 5], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.01, 0.01, 0.02]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 定义颜色列表
colors = [
    '#F44336',  # Very Light Red
    '#FB8C00',  # Light Red
    '#FDD835',  # Light Pink Red
    '#43A047',  # Soft Red
    '#1E88E5',  # Warm Light Red
    '#8E24AA',  # Bright Red
    '#F06292',  # Strong Red
    '#FBC02D',
    '#FFAB91',
    '#00ACC1',  # Pure Red
    '#D81B60',  # Dark Red
    '#00796B',  # Darker Red
    '#6D4C41',  # Deep Red
]

# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None
# Lists for evaluation metrics
val_accuracy_scores1 = []
val_auc_scores1 = []
val_precision_scores1 = []
val_specificity_scores1 = []
val_sensitivity_scores1 = []
val_npv_scores1 = []
val_ppv_scores1 = []
val_recall_scores1 = []
val_f1_scores1 = []
val_fpr_scores1 = []
val_rmse_scores1 = []
val_r2_scores1 = []
val_mae_scores1 = []
val_tn_scores1 = []
val_fp_scores1 = []
val_fn_scores1 = []
val_tp_scores1 = []
val_lift_scores1 = []
val_brier_scores1 = []
val_kappa_scores1 = []

# Fit models and plot ROC curve for external validation set
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # Predict probabilities on external validation set
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob1 = best_model_temp.predict_proba(val_features1)[:, 1]
    else:
        y_val_pred_prob1 = best_model_temp.decision_function(val_features1)

    # Calculate AUC
    auc = roc_auc_score(val_target1, y_val_pred_prob1)

    # Update best model if current model has higher AUC
    if auc > best_auc:
        best_auc1 = auc
        best_model_name = name
        best_model = best_model_temp

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(val_target1, y_val_pred_prob1)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # Calculate other evaluation metrics
    y_val_pred1 = best_model_temp.predict(val_features1)
    val_accuracy1 = accuracy_score(val_target1, y_val_pred1)
    val_precision1 = precision_score(val_target1, y_val_pred1)
    val_cm1 = confusion_matrix(val_target1, y_val_pred1)
    val_tn1, val_fp1, val_fn1, val_tp1 = val_cm1.ravel()
    val_specificity1 = val_tn1 / (val_tn1 + val_fp1)
    val_sensitivity1 = recall_score(val_target1, y_val_pred1)
    val_npv1 = val_tn1 / (val_tn1 + val_fn1)
    val_ppv1 = val_tp1 / (val_tp1 + val_fp1)
    val_recall1 = val_sensitivity1
    val_f11 = f1_score(val_target1, y_val_pred1)
    val_fpr1 = val_fp1 / (val_fp1 + val_tn1)
    val_rmse1 = mean_squared_error(val_target1, y_val_pred_prob1, squared=False)
    val_r21 = r2_score(val_target1, y_val_pred_prob1)
    val_mae1 = mean_absolute_error(val_target1, y_val_pred_prob1)
    val_kappa1 = cohen_kappa_score(val_target1, y_val_pred1)
    val_lift1 = average_precision_score(val_target1, y_val_pred_prob1) / (sum(val_target1) / len(val_target1))
    val_brier1 = brier_score_loss(val_target1, y_val_pred_prob1)

    # Append evaluation metrics to lists
    val_accuracy_scores1.append(val_accuracy1)
    val_auc_scores1.append(auc)
    val_precision_scores1.append(val_precision1)
    val_specificity_scores1.append(val_specificity1)
    val_sensitivity_scores1.append(val_sensitivity1)
    val_npv_scores1.append(val_npv1)
    val_ppv_scores1.append(val_ppv1)
    val_recall_scores1.append(val_recall1)
    val_f1_scores1.append(val_f11)
    val_fpr_scores1.append(val_fpr1)
    val_rmse_scores1.append(val_rmse1)
    val_r2_scores1.append(val_r21)
    val_mae_scores1.append(val_mae1)
    val_tn_scores1.append(val_tn1)
    val_fp_scores1.append(val_fp1)
    val_fn_scores1.append(val_fn1)
    val_tp_scores1.append(val_tp1)
    val_lift_scores1.append(val_lift1)
    val_brier_scores1.append(val_brier1)
    val_kappa_scores1.append(val_kappa1)

plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.grid(color='lightgray', linestyle='-', linewidth=1)  # Background grid lines
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Pretracheal Lymph Node Metastasis(Validation Set1)')
plt.legend(loc='lower right')
# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure1-C3-Pretracheal-roc-val1_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# Print best model name and AUC
print(f"Best model: {best_model_name} with AUC = {best_auc}")

# Create DataFrame for external validation metrics
Ext_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': val_accuracy_scores1,
    'AUC': val_auc_scores1,
    'Precision': val_precision_scores1,
    'Specificity': val_specificity_scores1,
    'Sensitivity': val_sensitivity_scores1,
    'Negative Predictive Value': val_npv_scores1,
    'Positive Predictive Value': val_ppv_scores1,
    'Recall': val_recall_scores1,
    'F1 Score': val_f1_scores1,
    'False Positive Rate': val_fpr_scores1,
    'RMSE': val_rmse_scores1,
    'R2': val_r2_scores1,
    'MAE': val_mae_scores1,
    'True Negatives': val_tn_scores1,
    'False Positives': val_fp_scores1,
    'False Negatives': val_fn_scores1,
    'True Positives': val_tp_scores1,
    'Lift': val_lift_scores1,
    'Brier Score': val_brier_scores1,
    'Kappa': val_kappa_scores1,  
})

# Display DataFrame
print(Ext_metrics_df)

# Export metrics to CSV
Ext_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/1.3.Pretracheal验证集1的评价指标.csv', index=False)


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
val_net_benefit1 = []

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob1 = best_model_temp.predict_proba(val_features1)[:, 1]
    else:
        y_val_pred_prob1 = best_model_temp.decision_function(val_features1)

    val_model_net_benefit1 = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        val_predictions1 = (y_val_pred_prob1 >= threshold).astype(int)
        tp = np.sum((val_target1 == 1) & (val_predictions1 == 1))
        fp = np.sum((val_target1 == 0) & (val_predictions1 == 1))
        fn = np.sum((val_target1 == 1) & (val_predictions1 == 0))
        tn = np.sum((val_target1 == 0) & (val_predictions1 == 0))
        
        net_benefit1 = (tp / len(val_target1)) - (fp / len(val_target1)) * (threshold / (1 - threshold))
        val_model_net_benefit1.append(net_benefit1)

    val_net_benefit1.append(val_model_net_benefit1)

# 转换为数组
val_net_benefit1 = np.array(val_net_benefit1)

# 计算所有人都进行干预时的净收益
val_all_predictions1 = np.ones_like(val_target1)  # 将所有预测标记为阳性（正类）
tp_all1 = np.sum((val_target1 == 1) & (val_all_predictions1 == 1))
fp_all1 = np.sum((val_target1 == 0) & (val_all_predictions1 == 1))

net_benefit_all1 = (tp_all1 / len(val_target1)) - (fp_all1 / len(val_target1)) * (thresholds / (1 - thresholds))
net_benefit_none1 = np.zeros_like(thresholds)

names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Extra Trees',
    'AdaBoost',
    'Gradient Boosting',
    'HistGradientBoosting',
    'XGBoost', 
    'CatBoost',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Neural Network',
    'Gaussian Naive Bayes',
]

# 绘制DCA曲线
for i in range(val_net_benefit1.shape[0]):
    plt.plot(thresholds, val_net_benefit1[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, net_benefit_none1, color='black', linestyle='-', label='None')
plt.plot(thresholds, net_benefit_all1, color='gray', linestyle='--', label='All')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
# 设置y轴的限制
plt.xlim(0, 0.6)
plt.ylim(-0.1, 0.35)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA For Pretracheal Lymph Node Metastasis(Validation Set1)')
plt.legend(loc='upper right')
# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure2-C3-Pretracheal-dca-val1_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()


# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 创建一个空列表来存储每个模型的校准曲线和Brier Score
val_calibration_curves1 = []
val_brier_scores1 = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob1 = best_model_temp.predict_proba(val_features1)[:, 1]
    else:
        y_val_pred_prob1 = best_model_temp.decision_function(val_features1)

    # 计算校准曲线
    val_fraction_of_positives1, val_mean_predicted_value1 = calibration_curve(val_target1, y_val_pred_prob1, n_bins=10)
    val_calibration_curves1.append((val_fraction_of_positives1, val_mean_predicted_value1, name, color))

    # 计算Brier分数
    val_brier_score = brier_score_loss(val_target1, y_val_pred_prob1)
    val_brier_scores1.append((name, val_brier_score))

    # 打印Brier分数
    print(f'{name} - Brier Score: {val_brier_score:.3f}')

# 绘制校准曲线和Brier Score
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in val_calibration_curves1:
    val_fraction_of_positives1, val_mean_predicted_value1, name, color = curve
    
    # 获取对应模型的Brier Score
    val_brier_score = next((score for model, score in val_brier_scores1 if model == name), None)
    
    # 将Brier Score赋予线颜色标注名称的后面
    if val_brier_score is not None:
        name += f' (Brier Score: {val_brier_score:.3f})'
    
    ax1.plot(val_mean_predicted_value1, val_fraction_of_positives1, "s-", label=name, color=color)
    
# 绘制"Perfectly calibrated"曲线
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Pretracheal Lymph Node Metastasis(Validation set1)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure3-C3-Pretracheal-CC-val1_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 初始化存储精确召回曲线和平均精确度的列表
val_precision_recall_curves1 = []
val_average_precision_scores1 = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob1 = best_model_temp.predict_proba(val_features1)[:, 1]
    else:
        y_val_pred_prob1 = best_model_temp.decision_function(val_features1)

    # 计算精确召回曲线
    val_precision1, val_recall1, _ = precision_recall_curve(val_target1, y_val_pred_prob1)
    val_average_precision1 = average_precision_score(val_target1, y_val_pred_prob1)

    # 存储结果
    val_precision_recall_curves1.append((val_precision1, val_recall1, f'{name} (AUPR: {val_average_precision1:.3f})', color))
    val_average_precision_scores1.append((f'{name} (AUPR: {val_average_precision1:.3f})', val_average_precision1))

    # 打印平均精确度
    print(f'{name} - Average Precision: {val_average_precision1:.3f}')

# 绘制精确召回曲线
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in val_precision_recall_curves1:
    val_precision1, val_recall1, name, color = curve
    ax2.plot(val_recall1, val_precision1, "-", color=color, label=name)

# 添加随机猜测曲线
ax2.plot([0, 1], [val_target1.mean(), val_target1.mean()], linestyle='--', color='black', label='Random Guessing')

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend(loc="lower left")
ax2.grid(True)

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Precision Recall Curves For Pretracheal Lymph Node Metastasis(Validation set1)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure4-C3-Pretracheal-PRC-val1_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier


# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')

# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR', ]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
    'Logistic Regression': (LogisticRegression(random_state=33), {'C': [0.01, 0.1, 1, 10, 100]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [1, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 70, 180], 'learning_rate': [0.01, 0.1, 0.2]}),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [10, 30, 50], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 100, 200], 'max_depth': [1, 3, 5], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5],}),
    'Support Vector Machine': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [30, 130, 150], 'max_depth': [1, 3, 5], }),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 定义颜色列表
colors = [
    '#F44336',  # Very Light Red
    '#FB8C00',  # Light Red
    '#FDD835',  # Light Pink Red
    '#43A047',  # Soft Red
    '#1E88E5',  # Warm Light Red
    '#8E24AA',  # Bright Red
    '#F06292',  # Strong Red
    '#FBC02D',
    '#FFAB91',
    '#00ACC1',  # Pure Red
    '#D81B60',  # Dark Red
    '#00796B',  # Darker Red
    '#6D4C41',  # Deep Red
]



# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None


# Lists for evaluation metrics
val_accuracy_scores2 = []
val_auc_scores2 = []
val_precision_scores2 = []
val_specificity_scores2 = []
val_sensitivity_scores2 = []
val_npv_scores2 = []
val_ppv_scores2 = []
val_recall_scores2 = []
val_f1_scores2 = []
val_fpr_scores2 = []
val_rmse_scores2 = []
val_r2_scores2 = []
val_mae_scores2 = []
val_tn_scores2 = []
val_fp_scores2 = []
val_fn_scores2 = []
val_tp_scores2 = []
val_lift_scores2 = []
val_brier_scores2 = []
val_kappa_scores2 = []

# Fit models and plot ROC curve for external validation set
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # Predict probabilities on external validation set
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob2= best_model_temp.predict_proba(val_features2)[:, 1]
    else:
        y_val_pred_prob2 = best_model_temp.decision_function(val_features2)

    # Calculate AUC
    auc = roc_auc_score(val_target2, y_val_pred_prob2)

    # Update best model if current model has higher AUC
    if auc > best_auc:
        best_auc1 = auc
        best_model_name = name
        best_model = best_model_temp

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(val_target2, y_val_pred_prob2)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # Calculate other evaluation metrics
    y_val_pred2 = best_model_temp.predict(val_features2)
    val_accuracy2 = accuracy_score(val_target2, y_val_pred2)
    val_precision2 = precision_score(val_target2, y_val_pred2)
    val_cm2 = confusion_matrix(val_target2, y_val_pred2)
    val_tn2, val_fp2, val_fn2, val_tp2 = val_cm2.ravel()
    val_specificity2 = val_tn2 / (val_tn2 + val_fp2)
    val_sensitivity2 = recall_score(val_target2, y_val_pred2)
    val_npv2 = val_tn2 / (val_tn2 + val_fn2)
    val_ppv2 = val_tp2 / (val_tp2 + val_fp2)
    val_recall2 = val_sensitivity2
    val_f12 = f1_score(val_target2, y_val_pred2)
    val_fpr2 = val_fp2 / (val_fp2 + val_tn2)
    val_rmse2 = mean_squared_error(val_target2, y_val_pred_prob2, squared=False)
    val_r22 = r2_score(val_target2, y_val_pred_prob2)
    val_mae2 = mean_absolute_error(val_target2, y_val_pred_prob2)
    val_kappa2 = cohen_kappa_score(val_target2, y_val_pred2)
    val_lift2 = average_precision_score(val_target2, y_val_pred_prob2) / (sum(val_target2) / len(val_target2))
    val_brier2 = brier_score_loss(val_target2, y_val_pred_prob2)

    # Append evaluation metrics to lists
    val_accuracy_scores2.append(val_accuracy2)
    val_auc_scores2.append(auc)
    val_precision_scores2.append(val_precision2)
    val_specificity_scores2.append(val_specificity2)
    val_sensitivity_scores2.append(val_sensitivity2)
    val_npv_scores2.append(val_npv2)
    val_ppv_scores2.append(val_ppv2)
    val_recall_scores2.append(val_recall2)
    val_f1_scores2.append(val_f12)
    val_fpr_scores2.append(val_fpr2)
    val_rmse_scores2.append(val_rmse2)
    val_r2_scores2.append(val_r22)
    val_mae_scores2.append(val_mae2)
    val_tn_scores2.append(val_tn2)
    val_fp_scores2.append(val_fp2)
    val_fn_scores2.append(val_fn2)
    val_tp_scores2.append(val_tp2)
    val_lift_scores2.append(val_lift2)
    val_brier_scores2.append(val_brier2)
    val_kappa_scores2.append(val_kappa2)

plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.grid(color='lightgray', linestyle='-', linewidth=1)  # Background grid lines
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Pretracheal Lymph Node Metastasis(Validation Set2)')
plt.legend(loc='lower right')
# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure1-C4-Pretracheal-roc-val2_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# Print best model name and AUC
print(f"Best model: {best_model_name} with AUC = {best_auc}")

# Create DataFrame for external validation metrics
Ext_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': val_accuracy_scores2,
    'AUC': val_auc_scores2,
    'Precision': val_precision_scores2,
    'Specificity': val_specificity_scores2,
    'Sensitivity': val_sensitivity_scores2,
    'Negative Predictive Value': val_npv_scores2,
    'Positive Predictive Value': val_ppv_scores2,
    'Recall': val_recall_scores2,
    'F1 Score': val_f1_scores2,
    'False Positive Rate': val_fpr_scores2,
    'RMSE': val_rmse_scores2,
    'R2': val_r2_scores2,
    'MAE': val_mae_scores2,
    'True Negatives': val_tn_scores2,
    'False Positives': val_fp_scores2,
    'False Negatives': val_fn_scores2,
    'True Positives': val_tp_scores2,
    'Lift': val_lift_scores2,
    'Brier Score': val_brier_scores2,
    'Kappa': val_kappa_scores2,  
})

# Display DataFrame
print(Ext_metrics_df)

# Export metrics to CSV
Ext_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/1.4.Pretracheal验证集2的评价指标.csv', index=False)


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
val_net_benefit2 = []

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob2 = best_model_temp.predict_proba(val_features2)[:, 1]
    else:
        y_val_pred_prob2 = best_model_temp.decision_function(val_features2)

    val_model_net_benefit2 = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        val_predictions2 = (y_val_pred_prob2 >= threshold).astype(int)
        tp = np.sum((val_target2 == 1) & (val_predictions2 == 1))
        fp = np.sum((val_target2 == 0) & (val_predictions2 == 1))
        fn = np.sum((val_target2 == 1) & (val_predictions2 == 0))
        tn = np.sum((val_target2 == 0) & (val_predictions2 == 0))
        
        net_benefit2 = (tp / len(val_target2)) - (fp / len(val_target2)) * (threshold / (1 - threshold))
        val_model_net_benefit2.append(net_benefit2)

    val_net_benefit2.append(val_model_net_benefit2)

# 转换为数组
val_net_benefit2 = np.array(val_net_benefit2)

# 计算所有人都进行干预时的净收益
val_all_predictions2 = np.ones_like(val_target2)  # 将所有预测标记为阳性（正类）
tp_all2 = np.sum((val_target2 == 1) & (val_all_predictions2 == 1))
fp_all2 = np.sum((val_target2 == 0) & (val_all_predictions2 == 1))

net_benefit_all2 = (tp_all2 / len(val_target2)) - (fp_all2 / len(val_target2)) * (thresholds / (1 - thresholds))
net_benefit_none2 = np.zeros_like(thresholds)

names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Extra Trees',
    'AdaBoost',
    'Gradient Boosting',
    'HistGradientBoosting',
    'XGBoost', 
    'CatBoost',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Neural Network',
    'Gaussian Naive Bayes',
]

# 绘制DCA曲线
for i in range(val_net_benefit2.shape[0]):
    plt.plot(thresholds, val_net_benefit2[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, net_benefit_none2, color='black', linestyle='-', label='None')
plt.plot(thresholds, net_benefit_all2, color='gray', linestyle='--', label='All')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
# 设置y轴的限制
plt.xlim(0, 0.6)
plt.ylim(-0.1, 0.28)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA For Pretracheal Lymph Node Metastasis(Validation Set2)')
plt.legend(loc='upper right')
# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure2-C4-Pretracheal-dca-val2_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 创建一个空列表来存储每个模型的校准曲线和Brier Score
val_calibration_curves2 = []
val_brier_scores2 = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob2 = best_model_temp.predict_proba(val_features2)[:, 1]
    else:
        y_val_pred_prob2 = best_model_temp.decision_function(val_features2)

    # 计算校准曲线
    val_fraction_of_positives2, val_mean_predicted_value2 = calibration_curve(val_target2, y_val_pred_prob2, n_bins=10)
    val_calibration_curves2.append((val_fraction_of_positives2, val_mean_predicted_value2, name, color))

    # 计算Brier分数
    val_brier_score2 = brier_score_loss(val_target2, y_val_pred_prob2)
    val_brier_scores2.append((name, val_brier_score2))

    # 打印Brier分数
    print(f'{name} - Brier Score: {val_brier_score2:.3f}')

# 绘制校准曲线和Brier Score
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in val_calibration_curves2:
    val_fraction_of_positives2, val_mean_predicted_value2, name, color = curve
    
    # 获取对应模型的Brier Score
    val_brier_score2 = next((score for model, score in val_brier_scores2 if model == name), None)
    
    # 将Brier Score赋予线颜色标注名称的后面
    if val_brier_score2 is not None:
        name += f' (Brier Score: {val_brier_score2:.3f})'
    
    ax2.plot(val_mean_predicted_value2, val_fraction_of_positives2, "s-", label=name, color=color)
    
# 绘制"Perfectly calibrated"曲线
ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
ax2.set_ylabel("Fraction of positives")
ax2.set_xlabel("Mean predicted value")
ax2.set_ylim([-0.05, 1.05])
ax2.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Pretracheal Lymph Node Metastasis(Validation set2)")
plt.tight_layout()

# 保存图像
formats = ['JPEG']
dpis = [1200]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure3-C4-Pretracheal-CC-val2_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 初始化存储精确召回曲线和平均精确度的列表
val_precision_recall_curves2 = []
val_average_precision_scores2 = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob2 = best_model_temp.predict_proba(val_features2)[:, 1]
    else:
        y_val_pred_prob2 = best_model_temp.decision_function(val_features2)

    # 计算精确召回曲线
    val_precision2, val_recall2, _ = precision_recall_curve(val_target2, y_val_pred_prob2)
    val_average_precision2 = average_precision_score(val_target2, y_val_pred_prob2)

    # 存储结果
    val_precision_recall_curves2.append((val_precision2, val_recall2, f'{name} (AUPR: {val_average_precision2:.3f})', color))
    val_average_precision_scores2.append((f'{name} (AUPR: {val_average_precision2:.3f})', val_average_precision2))

    # 打印平均精确度
    print(f'{name} - Average Precision: {val_average_precision2:.3f}')

# 绘制精确召回曲线
fig, ax3 = plt.subplots(figsize=(10, 6))

for curve in val_precision_recall_curves2:
    val_precision2, val_recall2, name, color = curve
    ax3.plot(val_recall2, val_precision2, "-", color=color, label=name)

# 添加随机猜测曲线
ax3.plot([0, 1], [val_target2.mean(), val_target2.mean()], linestyle='--', color='black', label='Random Guessing')

ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.set_ylim([0.0, 1.05])
ax3.set_xlim([0.0, 1.0])
ax3.legend(loc="lower left")
ax3.grid(True)

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Precision Recall Curves For Pretracheal Lymph Node Metastasis(Validation set2)")
plt.tight_layout()

# 保存图像
formats = ['tiff']
dpis = [300]
for fmt in formats:
    for dpi in dpis:
        plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure4-C4-Pretracheal-PRC-val2_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import roc_auc_score


# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')
# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
               ]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
   'Logistic Regression': (LogisticRegression(random_state=33), {'C': [0.01, 0.1, 1, 10, 100]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'max_depth': [1, 3, 4], 'min_samples_split': [1, 5,7]}),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost': (AdaBoostClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1],}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), {'max_iter': [50, 100, 200], 'max_depth': [1, 3, 5], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5],}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 定义颜色列表
colors = [
    '#F44336',  # Very Light Red
    '#FB8C00',  # Light Red
    '#FDD835',  # Light Pink Red
    '#43A047',  # Soft Red
    '#1E88E5',  # Warm Light Red
    '#8E24AA',  # Bright Red
    '#F06292',  # Strong Red
    '#FBC02D',
    '#FFAB91',
    '#00ACC1',  # Pure Red
    '#D81B60',  # Dark Red
    '#00796B',  # Darker Red
    '#6D4C41',  # Deep Red
]



def plot_learning_curve_with_external(estimator, title, X_train, y_train, X_test1, y_test1, X_test2, y_test2, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(f"Learning Curve For Pretracheal Lymph Node Metastasis: {title}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    validation1_scores = []
    validation2_scores = []
    for train_size in train_sizes:
        X_train_subset = X_train[:int(train_size)]
        y_train_subset = y_train[:int(train_size)]
        estimator.fit(X_train_subset, y_train_subset)
        if hasattr(estimator, 'predict_proba'):
            y_test_pred_prob1 = estimator.predict_proba(X_test1)[:, 1]
            y_test_pred_prob2 = estimator.predict_proba(X_test2)[:, 1]
        else:
            y_test_pred_prob1 = estimator.decision_function(X_test1)
            y_test_pred_prob2 = estimator.decision_function(X_test2)
        score1 = roc_auc_score(y_test1, y_test_pred_prob1)
        score2 = roc_auc_score(y_test2, y_test_pred_prob2)
        validation1_scores.append(score1)
        validation2_scores.append(score2)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="red")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="green")
    plt.fill_between(train_sizes, validation1_scores - np.std(validation1_scores), validation1_scores + np.std(validation1_scores), alpha=0.1, color="blue")
    plt.fill_between(train_sizes, validation2_scores - np.std(validation2_scores), validation2_scores + np.std(validation2_scores), alpha=0.1, color="skyblue")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label=f"Train score (mean ROC AUC={train_scores_mean[-1]:.3f})")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label=f"Test score (mean ROC AUC={test_scores_mean[-1]:.3f})")
    plt.plot(train_sizes, validation1_scores, 'o-', color="blue", label=f"Validation1 score (mean ROC AUC={np.mean(validation1_scores):.3f})")
    plt.plot(train_sizes, validation2_scores, 'o-', color="skyblue", label=f"Validation2 score (mean ROC AUC={np.mean(validation2_scores):.3f})")
    
    for i, train_size in enumerate(train_sizes):
        plt.text(train_size, train_scores_mean[i], f'{train_scores_mean[i]:.3f}', color='red')
        plt.text(train_size, test_scores_mean[i], f'{test_scores_mean[i]:.3f}', color='green')
        plt.text(train_size, validation1_scores[i], f'{validation1_scores[i]:.3f}', color='blue')
        plt.text(train_size, validation2_scores[i], f'{validation2_scores[i]:.3f}', color='skyblue')
    
    plt.legend(loc="best")

    formats = ['JPEG']
    dpis = [300]
    for fmt in formats:
        for dpi in dpis:
            plt.savefig(f'/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/2.补充材料2-筛选模型/Figure5-Pretracheal-LC_{title}_{dpi}dpi.{fmt}', format=fmt, dpi=dpi, bbox_inches='tight')
    plt.show()

# 绘制所有模型的学习曲线（训练集和两个外部验证集）
for name, (model, param_grid) in model_param_grid.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_
    
    plot_learning_curve_with_external(best_model_temp, name, class_x_tra, class_y_tra, val_features1, val_target1, val_features2, val_target2, cv=10, n_jobs=-1)

# %%
#模型筛选之PMRA--加上nomogram的版本绘制热图一体化
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier

# 加载数据
train_data = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
val_data1 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/2.V1Q编码后_插补矫正后.csv')
val_data2 = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/3.V2Q编码后_插补矫正后.csv')
# 提取特征和目标
feature_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
target_col = 'Pretracheal LNM'

train_features = train_data[feature_cols]
train_target = train_data[target_col]

val_features1 = val_data1[feature_cols]
val_target1 = val_data1[target_col]

val_features2 = val_data2[feature_cols]
val_target2 = val_data2[target_col]
# 数值变量标准化
num_cols = ['age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]
cat_cols = ['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',]

scaler = MinMaxScaler()
train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
val_features1[num_cols] = scaler.transform(val_features1[num_cols])
val_features2[num_cols] = scaler.transform(val_features2[num_cols])
# 分为训练集和验证集
class_x_tra, class_x_test, class_y_tra, class_y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=2)

# 定义模型和参数空间
model_param_grid = {
    'Logistic Regression': (LogisticRegression(random_state=33), {'C': [0.01, 0.1, 1, 10, 100]}),

    'Decision Tree': (DecisionTreeClassifier(random_state=33), {'max_depth': [5, 7, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(random_state=33), {'n_estimators': [50, 100, 150], 'max_depth': [1, 3, 4], 'min_samples_split': [1, 5,7]}),
    'Extra Trees': (ExtraTreesClassifier(random_state=33), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}),
    'AdaBoost':(XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=30), {'n_estimators': [50,200,300],'max_depth': [1,5,7],'learning_rate': [0.05, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],}), 
    'Gradient Boosting': (GradientBoostingClassifier(random_state=33), {}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=33), { }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=33), {'n_estimators': [1,300,800],'max_depth': [1,3,15],'learning_rate': [0.01, 0.1, 0.2],
                                                                                             'subsample': [0.5, 1.0],'gamma': [0, 0.1, 0.8],'colsample_bytree': [0.5, 0.7,1.2],
                                                                                             'min_child_weight': [1, 10],'scale_pos_weight': [1, 10],'reg_alpha': [0, 0.5],
                                                                                             'reg_lambda': [0,0.5]}), 
    'CatBoost': (CatBoostClassifier(random_state=33, silent=True), {'depth': [1, 3, 5], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Support Vector Machine': (SVC(probability=True, random_state=33), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [10, 20, 50],'weights': ['uniform'],'algorithm': ['ball_tree', 'kd_tree']}),
    'Neural Network': (MLPClassifier(random_state=33), {'hidden_layer_sizes': [(10,), (20,), (50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
}

# 进行交叉验证并存储所有模型的预测
model_results = {}
for name, (model, param_grid) in model_param_grid.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=8, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    if hasattr(best_model_temp, 'predict_proba'):
        y_train_pred_prob = best_model_temp.predict_proba(class_x_tra)[:, 1]
    else:
        y_train_pred_prob = best_model_temp.decision_function(class_x_tra)

    model_results[name] = y_train_pred_prob

# 加载R语言中计算的nomogram预测结果（从R中导出）
nomogram_results = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/1.R语言的结果/5.对比/3.Pretracheal.LNM.nomogram_predictions.csv')
model_results['Nomogram'] = nomogram_results['nomogram_prediction'].values



# PMRA实现
def pmra(models_results):
    pairwise_comparisons = list(itertools.combinations(models_results.keys(), 2))
    pmra_results = []

    for model_a, model_b in pairwise_comparisons:
        y_a = models_results[model_a]
        y_b = models_results[model_b]

        wins_a = np.sum(y_a > y_b)
        wins_b = np.sum(y_a < y_b)
        total_comparisons = len(y_a)

        prob_a_win = wins_a / total_comparisons
        prob_b_win = wins_b / total_comparisons

        model = sm.Logit(np.where(y_a > y_b, 1, 0), sm.add_constant(np.abs(y_a - y_b)))
        result = model.fit(disp=False)
        wald_statistic = result.tvalues[1] ** 2
        p_value = result.pvalues[1]

        pmra_results.append({
            'model_a': model_a,
            'model_b': model_b,
            'prob_a_win': prob_a_win,
            'prob_b_win': prob_b_win,
            'wald_statistic': wald_statistic,
            'p_value': p_value
        })

    return pmra_results

# 计算PMRA
pmra_results = pmra(model_results)

# 胜率和p值矩阵生成
models = list(set([res['model_a'] for res in pmra_results] + [res['model_b'] for res in pmra_results]))
prob_matrix = pd.DataFrame(np.zeros((len(models), len(models))), index=models, columns=models)
pvalue_matrix = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)

for res in pmra_results:
    prob_matrix.loc[res['model_a'], res['model_b']] = res['prob_a_win']
    prob_matrix.loc[res['model_b'], res['model_a']] = res['prob_b_win']
    pvalue_matrix.loc[res['model_a'], res['model_b']] = res['p_value']
    pvalue_matrix.loc[res['model_b'], res['model_a']] = res['p_value']

np.fill_diagonal(prob_matrix.values, 0.5)
np.fill_diagonal(pvalue_matrix.values, 1)
#--------------------图-----------------------------------------
# 假设 Random Forest 是基准模型
XGB_model = 'XGBoost'

# 提取与 Random Forest 相关的胜率（RF 行和列）
XGB_row = prob_matrix.loc[XGB_model, :]  # Random Forest 行的胜率
XGB_col = prob_matrix.loc[:, XGB_model]  # Random Forest 列的胜率

# 按照 RF 行和列的值对其他模型进行排序
sorted_models_by_XGB = XGB_row.sort_values(ascending=True).index  # 按行降序
sorted_models_by_XGB = [XGB_model] + [model for model in sorted_models_by_XGB if model != XGB_model]  # 保证 RF 在第一位

# 重新排序胜率概率矩阵和p值矩阵
prob_matrix = prob_matrix.loc[sorted_models_by_XGB, sorted_models_by_XGB]
pvalue_matrix = pvalue_matrix.loc[sorted_models_by_XGB, sorted_models_by_XGB]

# 绘制胜率概率热图
plt.figure(figsize=(10, 8))
sns.heatmap(prob_matrix, annot=True, cmap="Reds", cbar=True, fmt=".3f", linewidths=.5)
plt.title('Win Probability Heatmap For Pretracheal Lymph Node Metastasis')
plt.xlabel('model b')
plt.ylabel('model a')
plt.tight_layout()
prob_heatmap_XGB_path = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/4.补充材料3-模型对比/Figure1-C-Pretracheal-win_probability_heatmap.tiff'
plt.savefig(prob_heatmap_XGB_path, dpi=300, format='tiff', bbox_inches='tight')
plt.show()

# 绘制 Wald 检验的 p 值热图
plt.figure(figsize=(10, 8))
sns.heatmap(pvalue_matrix, annot=True, cmap="Blues", cbar=True, fmt=".3f", linewidths=.5)
plt.title('Wald Test p-value Heatmap For Pretracheal Lymph Node Metastasis')
plt.xlabel('model b')
plt.ylabel('model a')
plt.tight_layout()
pvalue_heatmap_XGB_path = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/4.补充材料3-模型对比/Figure2-C-Pretracheal-wald_pvalue_heatmap_all.tiff'
plt.savefig(pvalue_heatmap_XGB_path, dpi=300, format='tiff', bbox_inches='tight')
plt.show()
#-————————————————————————————————————————————————————————————-表--------------------------------------------
# 获取胜率排名，保留RF作为第一行
XGB_model = 'XGBoost'  # 假设模型名为 'Random Forest'
df_pmra_ranking = pd.DataFrame({
    'model': prob_matrix.index,
    'P. of Win against Top Model': prob_matrix[XGB_model].values,
    'Wald p-Value': pvalue_matrix[XGB_model].values
})

# 排序，RF 为第一行，其他模型按降序排列
remaining_models_sorted_by_win_prob_desc = df_pmra_ranking[df_pmra_ranking['model'] != XGB_model].sort_values(by='P. of Win against Top Model', ascending=False)['model'].tolist()
sorted_models_with_XGB_first = [XGB_model] + remaining_models_sorted_by_win_prob_desc
df_pmra_ranking = df_pmra_ranking.set_index('model').loc[sorted_models_with_XGB_first].reset_index()

# 添加序号列
df_pmra_ranking.reset_index(drop=True, inplace=True)
df_pmra_ranking.insert(0, '序号', df_pmra_ranking.index + 1)

# 保存到CSV
csv_path = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/4.补充材料3-模型对比/Table3-Pretracheal-pmra_ranking.csv'
df_pmra_ranking.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"PMRA ranking table saved to {csv_path}")


# %%
#SHAP
!pip install xgboost

!pip install shap

# %%
#3.1.1
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.01, 'n_estimators': 2, 'gamma': 0.7859285714285714, 'max_depth': 2, 'min_child_weight': 4,
                'colsample_bytree': 0.15000000000000002, 'colsample_bylevel': 0.0, 'subsample': 0.5789473684210527, 'reg_lambda': 0.1, 'reg_alpha': 0.20833333333333334,}
cv_params = {'n_estimators': np.linspace(0, 200, 3000, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.2
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 2, 'min_child_weight': 4,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.0, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'max_depth': np.linspace(1, 10, 20, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.3
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 4,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.0, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'min_child_weight': np.linspace(0, 10, 25, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.4
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.7859285714285714, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.0, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'gamma': np.linspace(0.001, 1, 15)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.5
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.0, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'subsample': np.linspace(0, 1, 20)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.6
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.0, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'colsample_bytree': np.linspace(0, 1, 21)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.7
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.30000000000000004, 'reg_alpha': 0.20833333333333334,}
cv_params = {'colsample_bylevel': np.linspace(0, 1, 21)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.8
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 0.20833333333333334,}
cv_params = {'reg_lambda': np.linspace(0, 1, 21)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.9
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667,}
cv_params = {'reg_alpha': np.linspace(0, 5, 25)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#3.1.9
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')
y = df['Pretracheal LNM']
X = df.drop(['Pretracheal LNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667,}
cv_params = {'eta': np.logspace(-2, 0, 10)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

# %%
#shap可视化图
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}

# Initialize and train the model
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Use SHAP to explain the model predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Sort features by SHAP value from largest to smallest
feature_order = np.argsort(mean_abs_shap)
sorted_features = X.columns[feature_order]
sorted_shap_values = shap_values[:, feature_order]

# Split data into metastasis (TPLNM=1) and non-metastasis (TPLNM=0)
X_metastasis = X[y == 1]
X_non_metastasis = X[y == 0]

# Calculate SHAP values for both groups
shap_values_metastasis = explainer.shap_values(X_metastasis)
shap_values_non_metastasis = explainer.shap_values(X_non_metastasis)

# Aggregating SHAP values
mean_shap_values_metastasis = np.abs(shap_values_metastasis).mean(axis=0)[feature_order]
mean_shap_values_non_metastasis = np.abs(shap_values_non_metastasis).mean(axis=0)[feature_order]

import numpy as np
import matplotlib.pyplot as plt

# 创建图形------------------------------------------------------------------条形图---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.8
index = np.arange(len(sorted_features))

# 绘制堆叠条形图
ax.barh(index, mean_shap_values_non_metastasis, bar_width, color='blue', label='Non-Metastasis (Pretracheal Lymph Node Metastasis=No)')
ax.barh(index, mean_shap_values_metastasis, bar_width, left=mean_shap_values_non_metastasis, color='red', label='Metastasis (Pretracheal Lymph Node Metastasis=Yes)')
# 设置标签和标题
ax.set_xlabel('Mean SHAP Value')
ax.set_title('Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Bar Plot')
ax.set_yticks(index)
ax.set_yticklabels(sorted_features)
# 添加图例
ax.legend()
# 调整布局以确保标签和图形不会重叠
plt.tight_layout()
# 保存图像为 TIFF 格式，设置 DPI 为 300
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C1-Pretracheal-combined_shap_bar_plot_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()


# 创建 SHAP summary plot（dot plot）----------------------------------------------汇总图--------------------------------------------------------------------
shap.summary_plot(sorted_shap_values, X.iloc[:, feature_order], plot_type="dot", show=False)
# 获取当前的绘图对象
plt.gca().set_title("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Summary Plot")
# 调整布局以确保图像显示正常
plt.tight_layout()
# 保存为 TIFF 格式，路径为指定位置，命名为 figure-5-a2
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C2-Pretracheal-summary_plot_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()

# SHAP interaction values plot-------------------------------------交互作用图------------------------------------
# 计算 SHAP 交互作用值
shap_interaction_values = explainer.shap_interaction_values(X)
# 创建更大的图形，以便标题和特征名称能够完整显示
plt.figure(figsize=(12, 8))
# 创建 SHAP 交互作用 summary plot
shap.summary_plot(shap_interaction_values, X.iloc[:, feature_order], plot_type="dot", show=False)
# 使用 suptitle 设置左对齐的标题，x=0 表示从最左侧开始
plt.suptitle("Feature Interaction Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Interaction Values Plot ", 
             x=0.01, y=0.98, ha='left', fontsize=12)
# 调整布局和边距，确保所有内容能显示完全
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局
plt.subplots_adjust(top=0.88)  # 调整顶部边距，确保标题不与图形内容重叠
# 保存为 TIFF 格式，路径为指定位置，命名为 Figure-5-a3
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C3-Pretracheal-interaction_plot_300dpi.jpg', format='JPEG', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()

# SHAP bar plot for all samples--------------------------------------------条形图------------------------------------------------------------
# 创建图形对象并调整尺寸
plt.figure(figsize=(12, 8))

# 生成 SHAP 条形图（bar plot）
shap.summary_plot(sorted_shap_values, X.iloc[:, feature_order], plot_type="bar", color="deeppink", show=False)

# 使用 suptitle 设置左对齐的标题，x=0 表示从最左侧开始
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Bar Plot", 
             x=0.01, y=0.98, ha='left', fontsize=12)
# 调整布局和边距，确保所有内容能显示完全
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局
plt.subplots_adjust(top=0.88)  # 调整顶部边距，确保标题不与图形内容重叠

# 保存为 TIFF 格式，路径为指定位置，命名为 Figure-5-a1.2
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C1.2-Pretracheal-bar_plot_300dpi.JPEG', format='JPEG', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 创建图形对象并调整尺寸-----------------------------------------------------决策曲线图----------------------------------------------------
plt.figure(figsize=(12, 8))

# 生成 SHAP 决策图
shap.decision_plot(explainer.expected_value, sorted_shap_values, X.iloc[:, feature_order], ignore_warnings=True, show=False)

# 使用 suptitle 设置左对齐的标题，x=0 表示从最左侧开始
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(All Samples)", 
             x=0.01, y=0.98, ha='left', fontsize=10)

# 调整布局和边距，确保所有内容能显示完全
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局
plt.subplots_adjust(top=0.88)  # 调整顶部边距，确保标题不与图形内容重叠

# 保存为 TIFF 格式，路径为指定位置，命名为 Figure-5-a4，使用 bbox_inches='tight' 确保内容不被截断
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C4-Pretracheal-decision_plot_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

#------------------------------------------随机20例的决策曲线图---------------------------------------------------------------
np.random.seed(33)

# 随机选择 20 个样本
sample_indices = np.random.choice(X_test.index, 20, replace=False)
sample_features = X_test.loc[sample_indices]

# 获取 SHAP 值
sample_shap_values = explainer.shap_values(sample_features)

# 绘制决策曲线图
plt.figure(figsize=(12, 8))
shap.decision_plot(explainer.expected_value, sample_shap_values, sample_features, ignore_warnings=True, show=False)

# 设置标题并左对齐
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(Random 20 Samples)", 
             x=0.01, y=0.98, ha='left', fontsize=10)

# 保存为 TIFF 格式，命名为 Figure5-a5
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.88)
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C5-Pretracheal-decision_plot_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()


# Predict and highlight misclassified samples-----------------------错误样本图------------------------------------------
# 预测和获取错误分类样本
y_pred = model.predict(sample_features)
misclassified = y_pred != y_test.loc[sample_indices]

# 绘制错误分类样本的决策曲线
plt.figure(figsize=(12, 8))
shap.decision_plot(explainer.expected_value, sample_shap_values, sample_features, highlight=misclassified, ignore_warnings=True, show=False)

# 设置标题并左对齐
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(Highlighted Misclassified Samples)", 
             x=0.01, y=0.98, ha='left', fontsize=10)

# 保存为 TIFF 格式，命名为 Figure5-a6
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.88)
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C6-Pretracheal-misclassified_decision_plot_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

import shap #---------------------------------全样本力导图------------------------------------------------------

# 假设 'explainer', 'shap_values', 和 'X' 已经定义

# 设置保存路径
save_path = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C7-Pretracheal-force_plot.html'

# 创建 SHAP 力导图并保存为 HTML
shap.save_html(save_path, shap.force_plot(explainer.expected_value, shap_values, X))

print(f"SHAP 力导图已保存为 {save_path}")



# 随机选择一个病例-------------------------------单个样本力导图------------------------------------------------------------
import shap
import matplotlib.pyplot as plt

# 假设 'explainer' 和 'X' 已经定义

# 随机选择一个病例
random_sample_index = np.random.choice(X.index, 1, replace=False)
random_sample = X.loc[random_sample_index]

# 获取该病例的 SHAP 值
random_sample_shap_values = explainer.shap_values(random_sample)

# 保存 HTML 文件路径
save_html_single_path = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C8-Pretracheal-force_plot_single_case.html'
shap.save_html(save_html_single_path, shap.force_plot(explainer.expected_value, random_sample_shap_values, random_sample))

# 生成 SHAP 力导图为 matplotlib 静态图
plt.figure(figsize=(12, 8))
shap_img_single = shap.force_plot(explainer.expected_value, random_sample_shap_values, random_sample, matplotlib=True, show=False)

# 保存为 TIFF 格式
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C8-Pretracheal-force_plot_single_case_300dpi.JPEG', format='JPEG', dpi=300, bbox_inches='tight')
plt.close()

print(f"SHAP 力导图已保存为 {save_html_single_path} (HTML) 和 TIFF 格式")

# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
from sklearn.model_selection import KFold
import sklearn
# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}



cv= KFold(n_splits=10, random_state=2, shuffle=True)
# 训练模型
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
    model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
    
# 创建解释器
explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value

# 固定选取的20例样本
features = X.iloc[1295:1315]
features_display = X.loc[features.index]
shap_values = explainer.shap_values(features)

# 决策图 - 固定选取的20例样本
plt.figure(figsize=(12, 8))
shap.decision_plot(expected_value, shap_values, features_display, feature_order=feature_order, show=False)
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(Random 20 Samples)", 
             x=0.01, y=0.98, ha='left', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.88)
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-c5.2-Pretracheal-decision_plot_random_20_samples_300dpi.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# 预测并找出错误分类的样本
y_pred = (shap_values.sum(1) + expected_value) > 0
misclassified = y_pred != y[1295:1315]

# 决策图 - 高亮显示错误分类的样本
plt.figure(figsize=(12, 8))
shap.decision_plot(expected_value, shap_values, features_display, highlight=misclassified, feature_order=feature_order, show=False)
plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(Highlighted Misclassified Samples)", 
             x=0.01, y=0.98, ha='left', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.88)
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C6.2-Pretracheal-decision_plot_misclassified_samples_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import shap
import matplotlib.pyplot as plt

# ----------------------------------------------------错误例示例----------------------------------------------------------

# 找出错误分类的样本
y_pred = (shap_values.sum(1) + expected_value) > 0
misclassified = y_pred != y[1295:1315]

# 找到错误分类的第一个样本
misclassified_indices = np.where(misclassified)[0]
if len(misclassified_indices) > 0:
    misclassified_sample_index = misclassified_indices[0]
    misclassified_features = features_display.iloc[[misclassified_sample_index]]
    misclassified_shap_values = shap_values[misclassified_sample_index]

    # 绘制并保存单个错误分类样本的决策图，并展示虚线
    plt.figure(figsize=(12, 8))
    shap.decision_plot(expected_value, misclassified_shap_values, misclassified_features, 
                       highlight=0, show=False)  # 使用 highlight=0 以虚线展示错误样本
    plt.suptitle("Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Decision Plot(Misclassified Sample)", 
                 x=0.01, y=0.98, ha='left', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.88)
    plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure5-C7-Pretracheal-decision_plot_misclassified_sample_300dpi.JPG', format='JPEG', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No misclassified samples found!")




# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}

# Train model
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# File paths
output_dir = '/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/'
titles = [
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-Heatmap Plot",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Bar Plot",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Bar Plot(Absolute Maximum Values)",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Beeswarm Plot",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Beeswarm Plot",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Bar Plot(Clustering)",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Bar Plot(Clustering-cutoff=0.8)",
    "Feature Importance by SHAP Values for Pretracheal Lymph Node Metastasis and Non-Metastasis-SHAP Bar Plot(Clustering-cutoff=1.8)"
]
filenames = [
    "Figure5-C9-Pretracheal-heatmap_300dpi.JPEG",
    "Figure5-C10-Pretracheal-bar_plot_300dpi.JPEG",
    "Figure5-C11-Pretracheal-bar_plot_abs_max_300dpi.JPEG",
    "Figure5-C12-Pretracheal-beeswarm_300dpi.JPEG",
    "Figure5-C13-Pretracheal-beeswarm_red_300dpi.JPEG",
    "Figure5-C14-Pretracheal-bar_plot_clustering_300dpi.JPEG",
    "Figure5-C15-Pretracheal-bar_plot_clustering_cutoff_0.8_300dpi.JPEG",
    "Figure5-C16-Pretracheal-bar_plot_clustering_cutoff_1.8_300dpi.JPEG"
]

# Plot and save each figure
plt.figure(figsize=(12, 8))

# Figure 1 - Heatmap
shap.plots.heatmap(shap_values[:1954], show=False)
plt.title(titles[0], fontsize=12, pad=20)  # 调整 pad 参数来增加标题与图像的间距
plt.savefig(output_dir + filenames[0], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2 - Bar plot
shap.plots.bar(shap_values, show=False)
plt.title(titles[1], fontsize=12)
plt.savefig(output_dir + filenames[1], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3 - Bar plot (absolute max values)
shap.plots.bar(shap_values.abs.max(0), show=False)
plt.title(titles[2], fontsize=12)
plt.savefig(output_dir + filenames[2], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4 - Beeswarm plot
shap.plots.beeswarm(shap_values, show=False)
plt.title(titles[3], fontsize=12)
plt.savefig(output_dir + filenames[3], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 5 - Beeswarm plot with red gradient
shap.plots.beeswarm(shap_values.abs, color="shap_red", show=False)
plt.title(titles[4], fontsize=12)
plt.savefig(output_dir + filenames[4], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 6 - Bar plot with clustering
clustering = shap.utils.hclust(X, y)
shap.plots.bar(shap_values, clustering=clustering, show=False)
plt.title(titles[5], fontsize=12)
plt.savefig(output_dir + filenames[5], format='eps', dpi=300, bbox_inches='tight')
plt.show()

# Figure 7 - Bar plot with clustering (cutoff=0.8)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.8, show=False)
plt.title(titles[6], fontsize=12)
plt.savefig(output_dir + filenames[6], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Figure 8 - Bar plot with clustering (cutoff=1.8)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=1.8, show=False)
plt.title(titles[7], fontsize=12)
plt.savefig(output_dir + filenames[7], format='JPEG', dpi=300, bbox_inches='tight')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.colors as mcolors

# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总H编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}

# Train model
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# Create a DataFrame for the feature importances
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# Calculate contribution percentage
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# Sort the DataFrame by contribution percentage (without changing the feature order)
shap_importance_df = shap_importance_df.sort_values(by='Contribution (%)', ascending=False)

# Create a color gradient from deep red to light red
cmap = mcolors.LinearSegmentedColormap.from_list("#C9A51A", ["#ECAC27", "#FFFF66"])

# Normalize the rank (not the values) for color mapping
ranks = np.arange(len(shap_importance_df))  # ranking from top to bottom based on the feature order
norm = plt.Normalize(ranks.min(), ranks.max())
colors = cmap(norm(ranks))

# Plot the feature contributions as a bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(shap_importance_df['Feature'], shap_importance_df['Contribution (%)'], color=colors)
plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Bar Plot of Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')
plt.gca().invert_yaxis()

# Add both SHAP mean and percentage values at the end of each bar
for bar, value, shap_value in zip(bars, shap_importance_df['Contribution (%)'], shap_importance_df['Mean Abs SHAP']):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{shap_value:.3f} ({value:.3f}%)', va='center')

# Save the full plot as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/Figure6-c1-Pretracheal-bar-plot.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# 1. 挑选前十个 SHAP 贡献值最大的特征
top_10_features = shap_importance_df.head(10).copy()

# 2. 重新计算前十个特征的贡献百分比
top_10_features['Contribution (%)'] = (top_10_features['Mean Abs SHAP'] / top_10_features['Mean Abs SHAP'].sum()) * 100

# 3. 创建颜色渐变（从深红到浅红）
cmap_top_10 = mcolors.LinearSegmentedColormap.from_list("#C9A51A", ["#ECAC27", "#FFFF66"])

# 排序新的前十个特征变量以进行颜色映射
ranks_top_10 = np.arange(len(top_10_features))
norm_top_10 = plt.Normalize(ranks_top_10.min(), ranks_top_10.max())
colors_top_10 = cmap_top_10(norm_top_10(ranks_top_10))  # 使用norm_top_10的__call__方法

# 4. 绘制前十个特征的贡献条形图
plt.figure(figsize=(12, 8))
bars_top_10 = plt.barh(top_10_features['Feature'], top_10_features['Contribution (%)'], color=colors_top_10)
plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')
plt.gca().invert_yaxis()

# 5. 在每个条形图后添加 SHAP 平均值和百分比
for bar, value, shap_value in zip(bars_top_10, top_10_features['Contribution (%)'], top_10_features['Mean Abs SHAP']):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{shap_value:.3f} ({value:.3f}%)', va='center')


# Save the top 10 plot as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C1.1-Pretracheal-top10-bar-plot.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}


# 训练模型
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# 计算SHAP值
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 计算每个特征的平均绝对SHAP值
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# 创建DataFrame存储特征重要性
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# 计算贡献百分比
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# 按照贡献百分比排序
shap_importance_df = shap_importance_df.sort_values(by='Contribution (%)', ascending=False)

# 颜色映射：深红到浅红渐变
cmap = mcolors.LinearSegmentedColormap.from_list("#C9A51A", ["#ECAC27","#FFFF66", ])

# 对贡献百分比进行归一化，生成颜色
norm = plt.Normalize(vmin=shap_importance_df['Contribution (%)'].min(), vmax=shap_importance_df['Contribution (%)'].max())
colors = cmap(norm(shap_importance_df['Contribution (%)']))

# 绘制泡泡图
plt.figure(figsize=(12, 8))

# 泡泡大小根据SHAP值调整
bubble_sizes = shap_importance_df['Mean Abs SHAP'] * 1200

# 添加网格线
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 绘制泡泡图，确保泡泡大小和SHAP值匹配，并反转Y轴
plt.scatter(shap_importance_df['Contribution (%)'], shap_importance_df['Feature'], 
            s=bubble_sizes, c=colors, alpha=0.8, edgecolors="w", linewidth=2)

plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Bubble Chart of Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')

# 反转Y轴，使得泡泡大的在上面
plt.gca().invert_yaxis()

# 在每个泡泡旁添加SHAP值和贡献百分比，确保位置正确
for i, (shap_value, contribution) in enumerate(zip(shap_importance_df['Mean Abs SHAP'], shap_importance_df['Contribution (%)'])):
    plt.text(contribution + 0.5, i, f'{shap_value:.3f} ({contribution:.3f}%)', va='center', fontsize=10)

# 保存全样本泡泡图
#plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C2-Pretracheal-bubble-plot2.eps', format='eps', dpi=300, bbox_inches='tight')
#plt.show()


# 选取前十个特征
top_10_df = shap_importance_df.head(10).copy()

# 重新计算前十个特征的贡献百分比
top_10_df['Contribution (%)'] = (top_10_df['Mean Abs SHAP'] / top_10_df['Mean Abs SHAP'].sum()) * 100

# 生成Top 10的颜色映射
ranks_top10 = np.arange(len(top_10_df))
norm_top10 = plt.Normalize(ranks_top10.min(), ranks_top10.max())
colors_top10 = cmap(norm_top10(ranks_top10))

# 泡泡大小根据SHAP值调整
bubble_sizes_top10 = top_10_df['Mean Abs SHAP'] * 1000

# 绘制Top 10泡泡图
plt.figure(figsize=(12, 8))

# 添加网格线
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 绘制Top 10泡泡图，确保Y轴反转
plt.scatter(top_10_df['Contribution (%)'], top_10_df['Feature'], 
            s=bubble_sizes_top10, c=colors_top10, alpha=0.8, edgecolors="w", linewidth=2)

plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Top 10 Bubble Chart of SHAP Values and Contribution Percentages')

# 反转Y轴，使得泡泡大的在上面
plt.gca().invert_yaxis()

# 为每个泡泡添加SHAP值和贡献度百分比
for i, (shap_value, contribution) in enumerate(zip(top_10_df['Mean Abs SHAP'], top_10_df['Contribution (%)'])):
    plt.text(contribution + 0.5, i, f'{shap_value:.3f} ({contribution:.3f}%)', va='center', fontsize=10)

# 保存Top 10泡泡图
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C2.2-Pretracheal-top10-bubble-plot.JPG', 
            format='JPEG', dpi=300, bbox_inches='tight')
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Sort shap_importance_df by 'Contribution (%)' in descending order
shap_importance_df_sorted = shap_importance_df.sort_values(by='Contribution (%)', ascending=False).reset_index(drop=True)

# Prepare data for pie chart
labels = shap_importance_df_sorted['Feature']
sizes = shap_importance_df_sorted['Contribution (%)']

# Reverse color mapping to match the contribution order (largest = darkest)
colors = cmap(np.linspace(0, 1, len(sizes)))  # Reverse the order of color mapping

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

plt.title('Pie Chart of Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')

# Save the pie chart as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C3-Pretracheal-pie-chart.jpg', format='JPEG', dpi=600, bbox_inches='tight')
plt.show()

# Top 10 pie chart
top_10_df_sorted = shap_importance_df_sorted.head(10)

# Prepare data for top 10 pie chart
labels_top10 = top_10_df_sorted['Feature']
sizes_top10 = top_10_df_sorted['Contribution (%)']

# Reverse color mapping for top 10
colors_top10 = cmap(np.linspace(0, 1, len(sizes_top10)))  # Ensure largest contribution has the darkest color

# Plot pie chart for top 10
plt.figure(figsize=(8, 8))
plt.pie(sizes_top10, labels=labels_top10, colors=colors_top10, autopct='%1.1f%%', startangle=140, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

plt.title('Top 10 Pie Chart of Importances and Contributions Percentage based on SHAP values for Pretracheal Central Lymph Node Metastasis XGBoost Model')

# Save the top 10 pie chart as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C3.2-Pretracheal-top10-pie-chart.jpg', format='JPEG', dpi=600, bbox_inches='tight')
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Sort shap_importance_df by 'Contribution (%)' in descending order
shap_importance_df_sorted = shap_importance_df.sort_values(by='Contribution (%)', ascending=False).reset_index(drop=True)

# Prepare data for donut chart (环图)
labels = shap_importance_df_sorted['Feature']
sizes = shap_importance_df_sorted['Contribution (%)']

# Reverse color mapping to match the contribution order (largest = darkest)
colors = cmap(np.linspace(0, 1, len(sizes)))  # Reverse the order of color mapping

# Plot donut chart (环图)
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

# Add a white circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Donut Chart of Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')

# Save the donut chart as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C4-Pretracheal-donut-chart.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Top 10 donut chart (环图)
top_10_df_sorted = shap_importance_df_sorted.head(10)

# Prepare data for top 10 donut chart
labels_top10 = top_10_df_sorted['Feature']
sizes_top10 = top_10_df_sorted['Contribution (%)']

# Reverse color mapping for top 10
colors_top10 = cmap(np.linspace(0, 1, len(sizes_top10)))  # Ensure largest contribution has the darkest color

# Plot donut chart for top 10
plt.figure(figsize=(8, 8))
plt.pie(sizes_top10, labels=labels_top10, colors=colors_top10, autopct='%1.1f%%', startangle=140, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

# Add a white circle at the center to make it a donut chart
centre_circle_top10 = plt.Circle((0, 0), 0.70, fc='white')
fig_top10 = plt.gcf()
fig_top10.gca().add_artist(centre_circle_top10)

plt.title('Top 10 Donut Chart of Importances and Contributions Percentage based on SHAP values for Pretracheal Lymph Node Metastasis XGBoost Model')

# Save the top 10 donut chart as a TIFF image with 300 dpi
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C4.2-Pretracheal-top10-donut-chart.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.colors as mcolors

# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/1.补充材料1-筛选变量/0.数据/1.总Q编码后_插补矫正后.csv')

# Select features and target variable
X = df[['Tumor border','Calcification','Side of position','Prelaryngeal LNM','Paratracheal LNM','Tumor Peripheral blood flow','Mulifocality',
                'age','bmi','size','Paratracheal LNMR','Paratracheal NLNM','Prelaryngeal LNMR','Con-Paratracheal LNMR',]]
y = df['Pretracheal LNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=2)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 159, 'gamma': 0.001, 'max_depth': 4, 'min_child_weight': 0,
                'colsample_bytree': 1.0, 'colsample_bylevel': 0.35000000000000003, 'subsample': 0.8421052631578947, 'reg_lambda': 0.5, 'reg_alpha': 1.0416666666666667, 'seed': 33}


model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# Create a DataFrame for the feature importances
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# Calculate contribution percentage
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# Sort the DataFrame by contribution percentage
shap_importance_df = shap_importance_df.sort_values(by='Contribution (%)', ascending=False).reset_index(drop=True)

# Ensure DataFrame has no missing values
X_train_df = pd.DataFrame(X_train, columns=feature_names).fillna(0)

# Sort columns in X_train based on SHAP importance
X_train_sorted = X_train_df[shap_importance_df['Feature']]

# Plot heatmap of all features based on SHAP importance order
plt.figure(figsize=(12, 8))
sns.heatmap(X_train_sorted.corr(), annot=True, cmap='coolwarm', linewidths=0.5, center=0)
plt.title('Pretracheal Lymph Node Metastasis Heatmap of All Features (Sorted by SHAP Importance)')
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C5-Pretracheal-SHAP-heatmap-all.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()

# Plot heatmap of top 10 features
top_10_features = shap_importance_df['Feature'].head(10)
X_train_top10_sorted = X_train_df[top_10_features]

plt.figure(figsize=(12, 8))
sns.heatmap(X_train_top10_sorted.corr(), annot=True, cmap='coolwarm', linewidths=0.5, center=0)
plt.title('Pretracheal Lymph Node Metastasis Heatmap of Top 10 Features (Sorted by SHAP Importance)')
plt.savefig('/Users/zj/Desktop/4.机器学习新/5.ptmc/2.结果/2.py的结果/3.SHAP可视化/figure6-C5-Pretracheal-SHAP-heatmap-top10.jpg', format='JPEG', dpi=300, bbox_inches='tight')
plt.show()


# %%



