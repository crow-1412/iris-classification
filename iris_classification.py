import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.model_selection import learning_curve
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve

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

def main():
    # 加载数据
    X, y = load_data('iris.data')
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建模型
    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 存储结果
    results = []
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        accuracy, train_time = train_and_evaluate(
            model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Training Time': train_time
        })
    
    # 将结果转换为DataFrame
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

    # 在main函数中添加
    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(10,), (10,5), (15,10)],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
    }

if __name__ == "__main__":
    main() 