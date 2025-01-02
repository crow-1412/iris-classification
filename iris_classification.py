import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os

# 创建figures目录（如果不存在）
if not os.path.exists('figures'):
    os.makedirs('figures')

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values   # 标签
    return X, y

# 模型训练和评估函数
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 记录训练时间
    train_time = time.time() - start_time
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 打印详细报告
    print(f"\n{model_name} Model Evaluation Results:")
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 添加交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    try:
        # 添加混淆矩阵可视化
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join('figures', f'{model_name}_confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {str(e)}")
    
    return accuracy, train_time

def plot_learning_curve(model, X, y, model_name):
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
        plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation Score')
        plt.title(f'{model_name} Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join('figures', f'{model_name}_learning_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create learning curve plot: {str(e)}")

def optimize_hyperparameters(model, X, y, param_grid, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"\n{model_name} Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def analyze_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, importances)
        plt.title(f'{model_name} Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join('figures', f'{model_name}_feature_importance.png'))
        plt.close()

def plot_validation_curve(model, X, y, param_name, param_range, model_name):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores.mean(axis=1), label='Training Score')
    plt.plot(param_range, test_scores.mean(axis=1), label='Validation Score')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'{model_name} Validation Curve')
    plt.legend()
    plt.savefig(os.path.join('figures', f'{model_name}_validation_curve.png'))
    plt.close()

def analyze_data(X, y):
    plt.figure(figsize=(10, 6))
    for i in range(X.shape[1]):
        plt.subplot(2, 2, i+1)
        for target in np.unique(y):
            plt.hist(X[y==target, i], alpha=0.5, label=target)
        plt.title(f'Feature {i+1} Distribution')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'data_distribution.png'))
    plt.close()

def analyze_data_distribution(X, y, feature_names):
    """数据分布可视化"""
    plt.figure(figsize=(12, 8))
    for i in range(X.shape[1]):
        plt.subplot(2, 2, i+1)
        for target in np.unique(y):
            plt.hist(X[y==target, i], alpha=0.5, label=f'Class {target}', bins=15)
        plt.title(f'{feature_names[i]} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'data_distribution.png'))
    plt.close()
    
    # 添加特征相关性分析
    plt.figure(figsize=(8, 6))
    df = pd.DataFrame(X, columns=feature_names)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join('figures', 'feature_correlation.png'))
    plt.close()

def feature_importance_analysis(models, X, feature_names):
    """特征重要性分析"""
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(8, 6))
            importances = model.feature_importances_
            plt.bar(feature_names, importances)
            plt.title(f'{model_name} Feature Importance')
            plt.xticks(rotation=45)
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join('figures', f'{model_name}_feature_importance.png'))
            plt.close()
        elif model_name == 'SVM' and hasattr(model, 'coef_'):
            plt.figure(figsize=(8, 6))
            importances = np.abs(model.coef_).mean(axis=0)
            plt.bar(feature_names, importances)
            plt.title(f'{model_name} Feature Importance')
            plt.xticks(rotation=45)
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join('figures', f'{model_name}_feature_importance.png'))
            plt.close()

def plot_learning_curves(models, X, y):
    """学习曲线分析"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    for model_name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            train_sizes=train_sizes,
            cv=5,
            n_jobs=-1
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
        plt.title(f'{model_name} Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join('figures', f'{model_name}_learning_curve.png'))
        plt.close()

def plot_roc_curves(models, X_test, y_test):
    """ROC曲线分析"""
    # 将标签转换为二进制格式
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            
            # 计算每个类的ROC曲线和AUC
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (class {i}, AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join('figures', 'roc_curves.png'))
    plt.close()

def main():
    # 加载数据
    X, y = load_data('iris.data')
    
    # 定义特征名称
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    
    # 数据分布可视化
    analyze_data_distribution(X, y, feature_names)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建模型
    models = {
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(10, 5),
            max_iter=2000,
            random_state=42,
            learning_rate_init=0.01,
            solver='adam',
            activation='relu',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            tol=1e-4,
            verbose=False
        ),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 训练和评估每个模型
    results = []
    for model_name, model in models.items():
        accuracy, train_time = train_and_evaluate(
            model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Training Time': train_time
        })
    
    # 特征重要性分析
    feature_importance_analysis(models, X_train_scaled, feature_names)
    
    # 学习曲线分析
    plot_learning_curves(models, X_train_scaled, y_train)
    
    # ROC曲线分析
    plot_roc_curves(models, X_test_scaled, y_test)
    
    # 将结果转换为DataFrame并打印
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Summary:")
    print(results_df)
    
    # 绘制比较图
    plt.figure(figsize=(12, 5))
    
    # 准确率比较
    plt.subplot(1, 2, 1)
    plt.bar(results_df['Model'], results_df['Accuracy'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=15)
    
    # 训练时间比较
    plt.subplot(1, 2, 2)
    plt.bar(results_df['Model'], results_df['Training Time'])
    plt.title('Model Training Time Comparison')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 