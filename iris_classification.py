import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import matplotlib.pyplot as plt

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
    print(f"\n{model_name} 模型评估结果:")
    print(f"训练时间: {train_time:.4f} 秒")
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, train_time

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
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 