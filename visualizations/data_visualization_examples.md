# 数据可视化示例

本文档提供了教程中各种数据可视化的代码示例和最佳实践，帮助读者创建专业的数据图表和分析报告。

## 📊 基础数据可视化

### 1. 数据分布可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DataVisualization:
    """数据可视化工具类"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_distribution(self, data, title="数据分布图"):
        """绘制数据分布图"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 直方图
        axes[0, 0].hist(data, bins=30, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('直方图')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频次')
        
        # 密度图
        axes[0, 1].hist(data, bins=30, density=True, alpha=0.7, color=self.colors[1])
        x = np.linspace(data.min(), data.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, data.mean(), data.std()), 
                       'r-', linewidth=2, label='正态分布拟合')
        axes[0, 1].set_title('密度图')
        axes[0, 1].legend()
        
        # 箱线图
        axes[1, 0].boxplot(data)
        axes[1, 0].set_title('箱线图')
        axes[1, 0].set_ylabel('数值')
        
        # Q-Q图
        stats.probplot(data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df, title="相关性矩阵"):
        """绘制相关性矩阵热力图"""
        plt.figure(figsize=self.figsize)
        correlation_matrix = df.corr()
        
        # 创建遮罩，只显示下三角
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_time_series(self, dates, values, title="时间序列图"):
        """绘制时间序列图"""
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # 原始时间序列
        axes[0].plot(dates, values, color=self.colors[0], linewidth=2)
        axes[0].set_title(f'{title} - 原始数据')
        axes[0].set_ylabel('数值')
        axes[0].grid(True, alpha=0.3)
        
        # 移动平均
        window = min(30, len(values) // 10)
        if window > 1:
            moving_avg = pd.Series(values).rolling(window=window).mean()
            axes[1].plot(dates, values, alpha=0.3, color=self.colors[0], label='原始数据')
            axes[1].plot(dates, moving_avg, color=self.colors[1], linewidth=2, 
                        label=f'{window}期移动平均')
            axes[1].set_title(f'{title} - 趋势分析')
            axes[1].set_ylabel('数值')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    
    # 创建可视化对象
    viz = DataVisualization()
    
    # 绘制分布图
    fig1 = viz.plot_distribution(data, "样本数据分布分析")
    plt.show()
    
    # 创建示例DataFrame
    df = pd.DataFrame({
        '特征1': np.random.normal(0, 1, 100),
        '特征2': np.random.normal(0, 1, 100),
        '特征3': np.random.normal(0, 1, 100),
        '特征4': np.random.normal(0, 1, 100)
    })
    df['特征5'] = df['特征1'] * 0.8 + np.random.normal(0, 0.2, 100)
    
    # 绘制相关性矩阵
    fig2 = viz.plot_correlation_matrix(df, "特征相关性分析")
    plt.show()
```

### 2. 机器学习结果可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import numpy as np

class MLVisualization:
    """机器学习结果可视化"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="混淆矩阵"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 添加准确率信息
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.15, 0.02, f'总体准确率: {accuracy:.3f}', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curves(self, y_true_list, y_scores_list, labels, title="ROC曲线对比"):
        """绘制多个模型的ROC曲线"""
        plt.figure(figsize=self.figsize)
        
        for i, (y_true, y_scores, label) in enumerate(zip(y_true_list, y_scores_list, labels)):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)],
                    linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_learning_curves(self, estimator, X, y, title="学习曲线"):
        """绘制学习曲线"""
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=self.figsize)
        
        plt.plot(train_sizes, train_mean, 'o-', color=self.colors[0],
                label='训练分数', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color=self.colors[0])
        
        plt.plot(train_sizes, val_mean, 'o-', color=self.colors[1],
                label='验证分数', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color=self.colors[1])
        
        plt.xlabel('训练样本数')
        plt.ylabel('分数')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names, importance_scores, 
                              title="特征重要性", top_k=20):
        """绘制特征重要性图"""
        # 排序并选择top_k个特征
        indices = np.argsort(importance_scores)[::-1][:top_k]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        plt.figure(figsize=(10, max(6, len(sorted_features) * 0.3)))
        
        # 水平条形图
        y_pos = np.arange(len(sorted_features))
        plt.barh(y_pos, sorted_scores, color=self.colors[0], alpha=0.7)
        
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('重要性分数')
        plt.title(title)
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        
        # 添加数值标签
        for i, score in enumerate(sorted_scores):
            plt.text(score + 0.01 * max(sorted_scores), i, f'{score:.3f}',
                    va='center', fontsize=9)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_model_comparison(self, model_names, metrics_dict, title="模型性能对比"):
        """绘制模型性能对比图"""
        metrics = list(metrics_dict.keys())
        n_metrics = len(metrics)
        n_models = len(model_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = metrics_dict[metric]
            x_pos = np.arange(n_models)
            
            bars = axes[i].bar(x_pos, values, color=self.colors[:n_models], alpha=0.7)
            axes[i].set_xlabel('模型')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric}对比')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(model_names, rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

# 示例使用
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                             n_redundant=10, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)
    
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    # 预测
    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    # 创建可视化对象
    ml_viz = MLVisualization()
    
    # 绘制混淆矩阵
    fig1 = ml_viz.plot_confusion_matrix(y_test, rf_pred, 
                                       labels=['类别0', '类别1'], 
                                       title="随机森林混淆矩阵")
    plt.show()
    
    # 绘制ROC曲线对比
    fig2 = ml_viz.plot_roc_curves([y_test, y_test], [rf_proba, lr_proba],
                                 ['随机森林', '逻辑回归'], "模型ROC曲线对比")
    plt.show()
    
    # 绘制特征重要性
    feature_names = [f'特征_{i}' for i in range(X.shape[1])]
    fig3 = ml_viz.plot_feature_importance(feature_names, rf.feature_importances_,
                                         "随机森林特征重要性")
    plt.show()
```

### 3. 深度学习训练可视化

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class DeepLearningVisualization:
    """深度学习训练过程可视化"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.training_history = defaultdict(list)
    
    def log_metrics(self, epoch, **metrics):
        """记录训练指标"""
        self.training_history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.training_history[key].append(value)
    
    def plot_training_history(self, title="训练历史"):
        """绘制训练历史曲线"""
        if not self.training_history:
            print("没有训练历史数据")
            return None
        
        epochs = self.training_history['epoch']
        metrics = {k: v for k, v in self.training_history.items() if k != 'epoch'}
        
        # 分离训练和验证指标
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith('val_')}
        val_metrics = {k.replace('val_', ''): v for k, v in metrics.items() if k.startswith('val_')}
        
        n_metrics = len(train_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, train_values) in enumerate(train_metrics.items()):
            axes[i].plot(epochs, train_values, 'o-', color=self.colors[0],
                        label=f'训练{metric_name}', linewidth=2)
            
            if metric_name in val_metrics:
                val_values = val_metrics[metric_name]
                axes[i].plot(epochs, val_values, 'o-', color=self.colors[1],
                           label=f'验证{metric_name}', linewidth=2)
            
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name}变化曲线')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_loss_landscape(self, loss_surface, title="损失函数地形图"):
        """绘制损失函数地形图"""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(-2, 2, loss_surface.shape[0])
        y = np.linspace(-2, 2, loss_surface.shape[1])
        X, Y = np.meshgrid(x, y)
        
        surf = ax.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.8)
        ax.set_xlabel('参数1')
        ax.set_ylabel('参数2')
        ax.set_zlabel('损失值')
        ax.set_title(title)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig
    
    def plot_gradient_flow(self, named_parameters, title="梯度流动图"):
        """绘制梯度流动图"""
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in named_parameters:
            if param.grad is not None and "bias" not in name:
                layers.append(name.replace('.weight', ''))
                ave_grads.append(param.grad.abs().mean().cpu().item())
                max_grads.append(param.grad.abs().max().cpu().item())
        
        plt.figure(figsize=self.figsize)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, 
                color=self.colors[0], label="最大梯度")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5,
                color=self.colors[1], label="平均梯度")
        
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
        plt.xlabel("层")
        plt.ylabel("梯度值")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_attention_weights(self, attention_weights, input_tokens, output_tokens,
                             title="注意力权重可视化"):
        """绘制注意力权重热力图"""
        plt.figure(figsize=(max(8, len(input_tokens) * 0.5), 
                           max(6, len(output_tokens) * 0.5)))
        
        sns.heatmap(attention_weights, 
                   xticklabels=input_tokens,
                   yticklabels=output_tokens,
                   cmap='Blues', annot=True, fmt='.2f')
        
        plt.xlabel('输入词汇')
        plt.ylabel('输出词汇')
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_model_architecture(self, layer_info, title="模型架构图"):
        """绘制模型架构图"""
        fig, ax = plt.subplots(figsize=(12, max(8, len(layer_info) * 0.8)))
        
        y_positions = np.arange(len(layer_info))
        layer_names = [info['name'] for info in layer_info]
        layer_params = [info['params'] for info in layer_info]
        
        # 绘制层级结构
        for i, (y_pos, name, params) in enumerate(zip(y_positions, layer_names, layer_params)):
            # 绘制矩形表示层
            rect_width = np.log10(params + 1) * 0.5  # 根据参数数量调整宽度
            rect = plt.Rectangle((0, y_pos - 0.3), rect_width, 0.6, 
                               facecolor=self.colors[i % len(self.colors)], 
                               alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # 添加层名称和参数数量
            ax.text(rect_width + 0.1, y_pos, f'{name}\n({params:,} 参数)', 
                   va='center', fontsize=10)
            
            # 绘制连接线
            if i < len(layer_info) - 1:
                ax.arrow(rect_width/2, y_pos + 0.3, 0, 0.4, 
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.set_xlim(-0.5, max([np.log10(info['params'] + 1) * 0.5 for info in layer_info]) + 3)
        ax.set_ylim(-0.5, len(layer_info) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=16)
        
        # 添加总参数数量
        total_params = sum(info['params'] for info in layer_info)
        ax.text(0.02, 0.98, f'总参数数量: {total_params:,}', 
               transform=ax.transAxes, va='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

# 示例使用
if __name__ == "__main__":
    # 创建深度学习可视化对象
    dl_viz = DeepLearningVisualization()
    
    # 模拟训练过程
    for epoch in range(1, 51):
        # 模拟训练指标
        train_loss = 2.0 * np.exp(-epoch * 0.1) + 0.1 * np.random.random()
        val_loss = 2.2 * np.exp(-epoch * 0.08) + 0.15 * np.random.random()
        train_acc = 1 - np.exp(-epoch * 0.15) + 0.05 * np.random.random()
        val_acc = 1 - np.exp(-epoch * 0.12) + 0.08 * np.random.random()
        
        dl_viz.log_metrics(epoch, 
                          loss=train_loss, 
                          accuracy=train_acc,
                          val_loss=val_loss, 
                          val_accuracy=val_acc)
    
    # 绘制训练历史
    fig1 = dl_viz.plot_training_history("模型训练过程")
    plt.show()
    
    # 生成示例损失地形图
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2) * np.exp(-(X**2 + Y**2)/2) + 0.1 * np.random.random((50, 50))
    
    fig2 = dl_viz.plot_loss_landscape(Z, "损失函数地形图")
    plt.show()
    
    # 示例模型架构
    layer_info = [
        {'name': '输入层', 'params': 0},
        {'name': '嵌入层', 'params': 50000},
        {'name': 'LSTM层1', 'params': 200000},
        {'name': 'LSTM层2', 'params': 150000},
        {'name': '全连接层', 'params': 10000},
        {'name': '输出层', 'params': 1000}
    ]
    
    fig3 = dl_viz.plot_model_architecture(layer_info, "LSTM模型架构")
    plt.show()
    
    # 示例注意力权重
    input_tokens = ['我', '喜欢', '机器', '学习']
    output_tokens = ['I', 'like', 'machine', 'learning']
    attention_weights = np.random.random((4, 4))
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig4 = dl_viz.plot_attention_weights(attention_weights, input_tokens, output_tokens,
                                        "中英翻译注意力权重")
    plt.show()
```

### 4. 业务指标可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BusinessMetricsVisualization:
    """业务指标可视化"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_kpi_dashboard(self, metrics_data, title="KPI仪表板"):
        """绘制KPI仪表板"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # 用户增长趋势
        axes[0, 0].plot(metrics_data['dates'], metrics_data['daily_users'], 
                       color=self.colors[0], linewidth=2)
        axes[0, 0].set_title('日活用户数')
        axes[0, 0].set_ylabel('用户数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 收入趋势
        axes[0, 1].bar(range(len(metrics_data['monthly_revenue'])), 
                      metrics_data['monthly_revenue'], 
                      color=self.colors[1], alpha=0.7)
        axes[0, 1].set_title('月度收入')
        axes[0, 1].set_ylabel('收入 (万元)')
        axes[0, 1].set_xticks(range(len(metrics_data['monthly_revenue'])))
        axes[0, 1].set_xticklabels([f'{i+1}月' for i in range(len(metrics_data['monthly_revenue']))])
        
        # 转化漏斗
        funnel_data = metrics_data['conversion_funnel']
        funnel_labels = list(funnel_data.keys())
        funnel_values = list(funnel_data.values())
        
        y_pos = np.arange(len(funnel_labels))
        axes[0, 2].barh(y_pos, funnel_values, color=self.colors[2], alpha=0.7)
        axes[0, 2].set_yticks(y_pos)
        axes[0, 2].set_yticklabels(funnel_labels)
        axes[0, 2].set_title('转化漏斗')
        axes[0, 2].set_xlabel('用户数')
        
        # 用户留存率
        retention_data = metrics_data['retention_rates']
        days = list(retention_data.keys())
        rates = list(retention_data.values())
        
        axes[1, 0].plot(days, rates, 'o-', color=self.colors[3], linewidth=2)
        axes[1, 0].set_title('用户留存率')
        axes[1, 0].set_xlabel('天数')
        axes[1, 0].set_ylabel('留存率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 用户行为分布
        behavior_data = metrics_data['user_behavior']
        axes[1, 1].pie(behavior_data.values(), labels=behavior_data.keys(), 
                      autopct='%1.1f%%', colors=self.colors[:len(behavior_data)])
        axes[1, 1].set_title('用户行为分布')
        
        # 模型性能指标
        model_metrics = metrics_data['model_performance']
        metric_names = list(model_metrics.keys())
        metric_values = list(model_metrics.values())
        
        bars = axes[1, 2].bar(metric_names, metric_values, 
                             color=self.colors[4], alpha=0.7)
        axes[1, 2].set_title('模型性能指标')
        axes[1, 2].set_ylabel('分数')
        axes[1, 2].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_ab_test_results(self, control_data, treatment_data, metric_name, 
                           title="A/B测试结果"):
        """绘制A/B测试结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 分布对比
        axes[0].hist(control_data, bins=30, alpha=0.7, 
                    color=self.colors[0], label='对照组')
        axes[0].hist(treatment_data, bins=30, alpha=0.7, 
                    color=self.colors[1], label='实验组')
        axes[0].set_xlabel(metric_name)
        axes[0].set_ylabel('频次')
        axes[0].set_title('分布对比')
        axes[0].legend()
        
        # 箱线图对比
        axes[1].boxplot([control_data, treatment_data], 
                       labels=['对照组', '实验组'])
        axes[1].set_ylabel(metric_name)
        axes[1].set_title('箱线图对比')
        
        # 统计摘要
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        improvement = (treatment_mean - control_mean) / control_mean * 100
        
        summary_text = f"""
        对照组均值: {control_mean:.3f}
        实验组均值: {treatment_mean:.3f}
        提升幅度: {improvement:.2f}%
        
        对照组标准差: {np.std(control_data):.3f}
        实验组标准差: {np.std(treatment_data):.3f}
        """
        
        axes[2].text(0.1, 0.5, summary_text, transform=axes[2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        axes[2].set_title('统计摘要')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_cohort_analysis(self, cohort_data, title="队列分析"):
        """绘制队列分析热力图"""
        plt.figure(figsize=self.figsize)
        
        # 创建热力图
        sns.heatmap(cohort_data, annot=True, fmt='.1%', cmap='YlOrRd',
                   cbar_kws={'label': '留存率'})
        
        plt.title(title)
        plt.xlabel('周期')
        plt.ylabel('队列')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_impact(self, feature_names, impact_scores, 
                          confidence_intervals, title="特征影响分析"):
        """绘制特征影响分析图"""
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        
        y_pos = np.arange(len(feature_names))
        
        # 绘制影响分数
        plt.barh(y_pos, impact_scores, color=self.colors[0], alpha=0.7)
        
        # 添加置信区间
        plt.errorbar(impact_scores, y_pos, 
                    xerr=[confidence_intervals[:, 0], confidence_intervals[:, 1]],
                    fmt='none', color='black', capsize=3)
        
        plt.yticks(y_pos, feature_names)
        plt.xlabel('影响分数')
        plt.title(title)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for i, score in enumerate(impact_scores):
            plt.text(score + 0.01 * max(abs(impact_scores)), i, f'{score:.3f}',
                    va='center', fontsize=9)
        
        plt.tight_layout()
        return plt.gcf()

# 示例使用
if __name__ == "__main__":
    # 生成示例业务数据
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    metrics_data = {
        'dates': dates,
        'daily_users': np.random.poisson(10000, 30) + np.arange(30) * 100,
        'monthly_revenue': [120, 135, 142, 158, 167, 180, 195, 210, 225, 240, 255, 270],
        'conversion_funnel': {
            '访问': 10000,
            '注册': 3000,
            '激活': 2000,
            '付费': 500,
            '留存': 300
        },
        'retention_rates': {1: 85, 7: 65, 14: 45, 30: 25, 60: 15, 90: 10},
        'user_behavior': {
            '浏览': 40,
            '搜索': 25,
            '购买': 20,
            '分享': 10,
            '其他': 5
        },
        'model_performance': {
            '准确率': 0.85,
            '精确率': 0.82,
            '召回率': 0.78,
            'F1分数': 0.80,
            'AUC': 0.88
        }
    }
    
    # 创建业务指标可视化对象
    biz_viz = BusinessMetricsVisualization()
    
    # 绘制KPI仪表板
    fig1 = biz_viz.plot_kpi_dashboard(metrics_data, "业务KPI仪表板")
    plt.show()
    
    # 生成A/B测试数据
    np.random.seed(42)
    control_data = np.random.normal(0.15, 0.05, 1000)  # 对照组转化率
    treatment_data = np.random.normal(0.18, 0.05, 1000)  # 实验组转化率
    
    fig2 = biz_viz.plot_ab_test_results(control_data, treatment_data, 
                                       "转化率", "转化率A/B测试")
    plt.show()
    
    # 生成队列分析数据
    cohort_data = pd.DataFrame({
        '第1周': [100, 85, 70, 60, 50],
        '第2周': [80, 68, 55, 45, 38],
        '第3周': [65, 52, 42, 35, 28],
        '第4周': [50, 40, 32, 26, 20]
    }, index=['1月队列', '2月队列', '3月队列', '4月队列', '5月队列'])
    cohort_data = cohort_data / 100  # 转换为百分比
    
    fig3 = biz_viz.plot_cohort_analysis(cohort_data, "用户留存队列分析")
    plt.show()
```

## 📈 交互式可视化

### 使用Plotly创建交互式图表

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class InteractiveVisualization:
    """交互式可视化工具"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_interactive_dashboard(self, data):
        """创建交互式仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间序列', '分布图', '散点图', '热力图'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {"type": "heatmap"}]]
        )
        
        # 时间序列图
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['values1'], 
                      name='指标1', line=dict(color=self.colors[0])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['values2'], 
                      name='指标2', line=dict(color=self.colors[1]),
                      yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # 分布图
        fig.add_trace(
            go.Histogram(x=data['distribution'], name='分布',
                        marker_color=self.colors[2], opacity=0.7),
            row=1, col=2
        )
        
        # 散点图
        fig.add_trace(
            go.Scatter(x=data['x_scatter'], y=data['y_scatter'],
                      mode='markers', name='散点',
                      marker=dict(color=data['colors'], 
                                 colorscale='Viridis',
                                 showscale=True)),
            row=2, col=1
        )
        
        # 热力图
        fig.add_trace(
            go.Heatmap(z=data['heatmap_data'], 
                      colorscale='RdYlBu_r',
                      showscale=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="交互式数据分析仪表板",
            showlegend=True,
            height=600
        )
        
        return fig
    
    def create_3d_scatter(self, x, y, z, color, title="3D散点图"):
        """创建3D散点图"""
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X轴',
                yaxis_title='Y轴',
                zaxis_title='Z轴'
            )
        )
        
        return fig
    
    def create_animated_plot(self, df, x_col, y_col, time_col, 
                           color_col=None, title="动画图表"):
        """创建动画图表"""
        fig = px.scatter(df, x=x_col, y=y_col, 
                        animation_frame=time_col,
                        color=color_col,
                        title=title,
                        range_x=[df[x_col].min()*0.9, df[x_col].max()*1.1],
                        range_y=[df[y_col].min()*0.9, df[y_col].max()*1.1])
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    dashboard_data = {
        'dates': dates,
        'values1': np.cumsum(np.random.randn(100)) + 100,
        'values2': np.cumsum(np.random.randn(100)) * 0.1 + 10,
        'distribution': np.random.normal(0, 1, 1000),
        'x_scatter': np.random.randn(200),
        'y_scatter': np.random.randn(200),
        'colors': np.random.randn(200),
        'heatmap_data': np.random.randn(10, 10)
    }
    
    # 创建交互式可视化对象
    interactive_viz = InteractiveVisualization()
    
    # 创建交互式仪表板
    dashboard_fig = interactive_viz.create_interactive_dashboard(dashboard_data)
    dashboard_fig.show()
    
    # 创建3D散点图
    x = np.random.randn(500)
    y = np.random.randn(500)
    z = np.random.randn(500)
    color = x + y + z
    
    scatter_3d_fig = interactive_viz.create_3d_scatter(x, y, z, color, "3D数据分布")
    scatter_3d_fig.show()
```

## 📋 可视化最佳实践

### 1. 颜色选择指南

- **定性数据**: 使用不同色相的颜色
- **定量数据**: 使用单一色相的渐变
- **对比数据**: 使用互补色
- **时间序列**: 使用连续的颜色映射

### 2. 图表类型选择

- **分布**: 直方图、密度图、箱线图
- **关系**: 散点图、相关性矩阵
- **比较**: 条形图、雷达图
- **趋势**: 折线图、面积图
- **组成**: 饼图、堆叠图

### 3. 交互设计原则

- **渐进式披露**: 从概览到细节
- **直接操作**: 点击、拖拽、缩放
- **即时反馈**: 实时更新和响应
- **上下文保持**: 保持用户的操作状态

### 4. 性能优化建议

- **数据采样**: 大数据集使用采样显示
- **延迟加载**: 按需加载详细数据
- **缓存策略**: 缓存计算结果
- **渲染优化**: 使用Canvas或WebGL

这些可视化示例和最佳实践为教程提供了全面的数据展示解决方案，帮助读者创建专业、美观且富有洞察力的数据图表。