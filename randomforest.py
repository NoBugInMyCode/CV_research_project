import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #随机森林模型
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
# 读取文件
file_path = "CW_project4.txt"
data = np.loadtxt(file_path)

# 提取最后一列作为y
y = data[:, -1]

# 修改标签，确保只包含0或1
y_binary = (y > 0).astype(int)

# 删除最后一列
x = data[:, :-1]

#归一化（可要可不要）
# norm = MinMaxScaler()
# x=norm.fit_transform(x)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, train_size=0.8, test_size=0.2)

# 随机森林模型（模型在这里改）
model = RandomForestClassifier(n_estimators=100, random_state=0, verbose=2, n_jobs=4)  # 100棵树，打印状态，4个核心并行
model.fit(X_train, y_train)  # 训练模型
preds = model.predict(X_test)  # 进行预测

# 计算准确率
accuracy = accuracy_score(y_test, preds)
print("准确率：", accuracy)

# 打印分类报告，包括精确度、召回率、F1分数等
report = classification_report(y_test, preds)
print("分类报告：\n", report)

# 计算 Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)

# 计算 ROC
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# 画 PR 曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

# 画 ROC 曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

#计算混淆矩阵
conf_matrix = confusion_matrix(y_test, preds)

# 画混淆矩阵
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ["Background", "Signal"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center", color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()