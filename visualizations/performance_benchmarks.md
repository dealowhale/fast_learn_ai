# 性能基准测试可视化

本文档提供了AI模型和系统性能基准测试的可视化方案，帮助开发者直观地比较和分析不同模型、算法和系统配置的性能表现。

## 🚀 模型性能对比

### 1. 模型准确率对比

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelPerformanceVisualizer:
    """模型性能可视化器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_accuracy_comparison(self, model_data):
        """创建模型准确率对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能全面对比分析', fontsize=16, fontweight='bold')
        
        # 1. 准确率条形图
        ax1 = axes[0, 0]
        models = list(model_data.keys())
        accuracies = [model_data[model]['accuracy'] for model in models]
        colors = sns.color_palette("viridis", len(models))
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('模型准确率对比', fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. 训练时间对比
        ax2 = axes[0, 1]
        training_times = [model_data[model]['training_time'] for model in models]
        
        bars2 = ax2.bar(models, training_times, color=colors, alpha=0.8)
        ax2.set_title('训练时间对比', fontweight='bold')
        ax2.set_ylabel('训练时间 (小时)')
        
        for bar, time in zip(bars2, training_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. 模型大小对比
        ax3 = axes[1, 0]
        model_sizes = [model_data[model]['model_size'] for model in models]
        
        bars3 = ax3.bar(models, model_sizes, color=colors, alpha=0.8)
        ax3.set_title('模型大小对比', fontweight='bold')
        ax3.set_ylabel('模型大小 (MB)')
        
        for bar, size in zip(bars3, model_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{size:.0f}MB', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. 推理速度对比
        ax4 = axes[1, 1]
        inference_speeds = [model_data[model]['inference_speed'] for model in models]
        
        bars4 = ax4.bar(models, inference_speeds, color=colors, alpha=0.8)
        ax4.set_title('推理速度对比', fontweight='bold')
        ax4.set_ylabel('推理速度 (samples/sec)')
        
        for bar, speed in zip(bars4, inference_speeds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{speed:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_radar_chart(self, model_data):
        """创建雷达图对比"""
        fig = go.Figure()
        
        # 定义评估维度
        categories = ['准确率', '训练速度', '推理速度', '内存效率', '可解释性']
        
        for model_name, data in model_data.items():
            # 标准化数据到0-100范围
            values = [
                data['accuracy'],
                100 - (data['training_time'] / max([d['training_time'] for d in model_data.values()]) * 100),
                data['inference_speed'] / max([d['inference_speed'] for d in model_data.values()]) * 100,
                100 - (data['model_size'] / max([d['model_size'] for d in model_data.values()]) * 100),
                data.get('interpretability', 50)  # 默认值
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="模型综合性能雷达图",
            title_x=0.5
        )
        
        return fig

# 示例数据
model_performance_data = {
    'BERT-Base': {
        'accuracy': 88.5,
        'training_time': 12.5,
        'model_size': 440,
        'inference_speed': 156,
        'interpretability': 70
    },
    'RoBERTa': {
        'accuracy': 90.2,
        'training_time': 15.8,
        'model_size': 498,
        'inference_speed': 142,
        'interpretability': 65
    },
    'DistilBERT': {
        'accuracy': 86.1,
        'training_time': 6.2,
        'model_size': 255,
        'inference_speed': 312,
        'interpretability': 75
    },
    'ALBERT': {
        'accuracy': 89.3,
        'training_time': 8.9,
        'model_size': 89,
        'inference_speed': 198,
        'interpretability': 68
    },
    'GPT-2': {
        'accuracy': 87.8,
        'training_time': 18.3,
        'model_size': 774,
        'inference_speed': 89,
        'interpretability': 45
    }
}

# 使用示例
visualizer = ModelPerformanceVisualizer()
fig1 = visualizer.create_accuracy_comparison(model_performance_data)
fig2 = visualizer.create_radar_chart(model_performance_data)
```

### 2. 训练过程可视化

```python
class TrainingProgressVisualizer:
    """训练过程可视化器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('default')
        sns.set_palette("Set2")
    
    def create_training_curves(self, training_history):
        """创建训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('模型训练过程全面监控', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # 1. 损失函数曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('损失函数变化', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, training_history['train_acc'], 'g-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, training_history['val_acc'], 'orange', label='验证准确率', linewidth=2)
        ax2.set_title('准确率变化', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        ax3 = axes[0, 2]
        ax3.plot(epochs, training_history['learning_rate'], 'purple', linewidth=2)
        ax3.set_title('学习率调度', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 梯度范数
        ax4 = axes[1, 0]
        ax4.plot(epochs, training_history['grad_norm'], 'brown', linewidth=2)
        ax4.set_title('梯度范数', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gradient Norm')
        ax4.grid(True, alpha=0.3)
        
        # 5. 训练时间
        ax5 = axes[1, 1]
        ax5.plot(epochs, training_history['epoch_time'], 'teal', linewidth=2)
        ax5.set_title('每轮训练时间', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Time (seconds)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 内存使用
        ax6 = axes[1, 2]
        ax6.plot(epochs, training_history['memory_usage'], 'navy', linewidth=2)
        ax6.set_title('内存使用情况', fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Memory (GB)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_training_dashboard(self, training_history):
        """创建交互式训练仪表板"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('损失函数', '准确率', '学习率', '梯度范数', '训练时间', '内存使用'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(training_history['train_loss']) + 1))
        
        # 损失函数
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_loss'], 
                      name='训练损失', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_loss'], 
                      name='验证损失', line=dict(color='red')),
            row=1, col=1
        )
        
        # 准确率
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_acc'], 
                      name='训练准确率', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_acc'], 
                      name='验证准确率', line=dict(color='orange')),
            row=1, col=2
        )
        
        # 学习率
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['learning_rate'], 
                      name='学习率', line=dict(color='purple')),
            row=2, col=1
        )
        
        # 梯度范数
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['grad_norm'], 
                      name='梯度范数', line=dict(color='brown')),
            row=2, col=2
        )
        
        # 训练时间
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['epoch_time'], 
                      name='训练时间', line=dict(color='teal')),
            row=3, col=1
        )
        
        # 内存使用
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['memory_usage'], 
                      name='内存使用', line=dict(color='navy')),
            row=3, col=2
        )
        
        fig.update_layout(
            height=900,
            title_text="模型训练过程交互式监控仪表板",
            title_x=0.5,
            showlegend=True
        )
        
        return fig

# 示例训练历史数据
training_history_data = {
    'train_loss': [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35],
    'val_loss': [2.1, 1.7, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.65, 0.6],
    'train_acc': [0.3, 0.45, 0.6, 0.72, 0.81, 0.87, 0.91, 0.94, 0.96, 0.97],
    'val_acc': [0.35, 0.48, 0.58, 0.68, 0.76, 0.82, 0.86, 0.89, 0.91, 0.92],
    'learning_rate': [0.001, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001],
    'grad_norm': [15.2, 12.8, 10.5, 8.9, 7.2, 5.8, 4.6, 3.9, 3.2, 2.8],
    'epoch_time': [120, 118, 115, 112, 110, 108, 106, 105, 104, 103],
    'memory_usage': [8.5, 8.3, 8.1, 7.9, 7.8, 7.6, 7.5, 7.4, 7.3, 7.2]
}

# 使用示例
training_viz = TrainingProgressVisualizer()
fig3 = training_viz.create_training_curves(training_history_data)
fig4 = training_viz.create_interactive_training_dashboard(training_history_data)
```

## 📊 系统性能基准测试

### 3. 硬件性能对比

```python
class HardwarePerformanceVisualizer:
    """硬件性能可视化器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("rocket")
    
    def create_gpu_comparison(self, gpu_data):
        """创建GPU性能对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GPU性能全面对比分析', fontsize=16, fontweight='bold')
        
        gpus = list(gpu_data.keys())
        
        # 1. 训练速度对比
        ax1 = axes[0, 0]
        training_speeds = [gpu_data[gpu]['training_speed'] for gpu in gpus]
        colors = sns.color_palette("viridis", len(gpus))
        
        bars1 = ax1.barh(gpus, training_speeds, color=colors, alpha=0.8)
        ax1.set_title('训练速度对比 (samples/sec)', fontweight='bold')
        ax1.set_xlabel('训练速度')
        
        for i, (bar, speed) in enumerate(zip(bars1, training_speeds)):
            width = bar.get_width()
            ax1.text(width + 50, bar.get_y() + bar.get_height()/2,
                    f'{speed}', ha='left', va='center', fontweight='bold')
        
        # 2. 内存容量对比
        ax2 = axes[0, 1]
        memory_sizes = [gpu_data[gpu]['memory_gb'] for gpu in gpus]
        
        bars2 = ax2.barh(gpus, memory_sizes, color=colors, alpha=0.8)
        ax2.set_title('显存容量对比 (GB)', fontweight='bold')
        ax2.set_xlabel('显存容量')
        
        for i, (bar, memory) in enumerate(zip(bars2, memory_sizes)):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{memory}GB', ha='left', va='center', fontweight='bold')
        
        # 3. 功耗对比
        ax3 = axes[1, 0]
        power_consumption = [gpu_data[gpu]['power_watts'] for gpu in gpus]
        
        bars3 = ax3.barh(gpus, power_consumption, color=colors, alpha=0.8)
        ax3.set_title('功耗对比 (Watts)', fontweight='bold')
        ax3.set_xlabel('功耗')
        
        for i, (bar, power) in enumerate(zip(bars3, power_consumption)):
            width = bar.get_width()
            ax3.text(width + 10, bar.get_y() + bar.get_height()/2,
                    f'{power}W', ha='left', va='center', fontweight='bold')
        
        # 4. 性价比分析
        ax4 = axes[1, 1]
        prices = [gpu_data[gpu]['price_usd'] for gpu in gpus]
        performance_per_dollar = [training_speeds[i] / prices[i] for i in range(len(gpus))]
        
        bars4 = ax4.barh(gpus, performance_per_dollar, color=colors, alpha=0.8)
        ax4.set_title('性价比对比 (性能/美元)', fontweight='bold')
        ax4.set_xlabel('性价比')
        
        for i, (bar, ratio) in enumerate(zip(bars4, performance_per_dollar)):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{ratio:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_scalability_analysis(self, scalability_data):
        """创建可扩展性分析"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('模型训练可扩展性分析', fontsize=14, fontweight='bold')
        
        # 1. GPU数量 vs 训练速度
        ax1 = axes[0]
        gpu_counts = scalability_data['gpu_counts']
        training_speeds = scalability_data['training_speeds']
        ideal_speeds = [training_speeds[0] * count for count in gpu_counts]
        
        ax1.plot(gpu_counts, training_speeds, 'bo-', linewidth=2, markersize=8, label='实际性能')
        ax1.plot(gpu_counts, ideal_speeds, 'r--', linewidth=2, alpha=0.7, label='理想线性扩展')
        ax1.set_title('GPU扩展性能分析', fontweight='bold')
        ax1.set_xlabel('GPU数量')
        ax1.set_ylabel('训练速度 (samples/sec)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 计算扩展效率
        efficiency = [training_speeds[i] / ideal_speeds[i] * 100 for i in range(len(gpu_counts))]
        
        # 2. 扩展效率
        ax2 = axes[1]
        bars = ax2.bar(gpu_counts, efficiency, color='green', alpha=0.7)
        ax2.set_title('扩展效率分析', fontweight='bold')
        ax2.set_xlabel('GPU数量')
        ax2.set_ylabel('扩展效率 (%)')
        ax2.set_ylim(0, 100)
        
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig

# 示例GPU数据
gpu_performance_data = {
    'RTX 4090': {
        'training_speed': 2850,
        'memory_gb': 24,
        'power_watts': 450,
        'price_usd': 1599
    },
    'RTX 4080': {
        'training_speed': 2100,
        'memory_gb': 16,
        'power_watts': 320,
        'price_usd': 1199
    },
    'RTX 3090': {
        'training_speed': 2200,
        'memory_gb': 24,
        'power_watts': 350,
        'price_usd': 999
    },
    'A100': {
        'training_speed': 3200,
        'memory_gb': 80,
        'power_watts': 400,
        'price_usd': 15000
    },
    'V100': {
        'training_speed': 1800,
        'memory_gb': 32,
        'power_watts': 300,
        'price_usd': 8000
    }
}

# 可扩展性数据
scalability_test_data = {
    'gpu_counts': [1, 2, 4, 8, 16],
    'training_speeds': [1000, 1900, 3600, 6800, 12500]
}

# 使用示例
hardware_viz = HardwarePerformanceVisualizer()
fig5 = hardware_viz.create_gpu_comparison(gpu_performance_data)
fig6 = hardware_viz.create_scalability_analysis(scalability_test_data)
```

### 4. 算法复杂度分析

```python
class AlgorithmComplexityVisualizer:
    """算法复杂度可视化器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('default')
        sns.set_palette("tab10")
    
    def create_complexity_comparison(self, algorithms_data):
        """创建算法复杂度对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('算法复杂度全面分析', fontsize=16, fontweight='bold')
        
        input_sizes = np.logspace(1, 6, 50)  # 10^1 到 10^6
        
        # 1. 时间复杂度对比
        ax1 = axes[0, 0]
        for algo_name, complexity_func in algorithms_data['time_complexity'].items():
            times = [complexity_func(n) for n in input_sizes]
            ax1.loglog(input_sizes, times, label=algo_name, linewidth=2)
        
        ax1.set_title('时间复杂度对比', fontweight='bold')
        ax1.set_xlabel('输入规模 (n)')
        ax1.set_ylabel('时间复杂度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 空间复杂度对比
        ax2 = axes[0, 1]
        for algo_name, complexity_func in algorithms_data['space_complexity'].items():
            spaces = [complexity_func(n) for n in input_sizes]
            ax2.loglog(input_sizes, spaces, label=algo_name, linewidth=2)
        
        ax2.set_title('空间复杂度对比', fontweight='bold')
        ax2.set_xlabel('输入规模 (n)')
        ax2.set_ylabel('空间复杂度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 实际运行时间测试
        ax3 = axes[1, 0]
        test_sizes = [100, 500, 1000, 5000, 10000]
        
        for algo_name, runtime_data in algorithms_data['actual_runtime'].items():
            ax3.plot(test_sizes, runtime_data, 'o-', label=algo_name, linewidth=2, markersize=6)
        
        ax3.set_title('实际运行时间测试', fontweight='bold')
        ax3.set_xlabel('数据集大小')
        ax3.set_ylabel('运行时间 (秒)')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 内存使用测试
        ax4 = axes[1, 1]
        for algo_name, memory_data in algorithms_data['actual_memory'].items():
            ax4.plot(test_sizes, memory_data, 's-', label=algo_name, linewidth=2, markersize=6)
        
        ax4.set_title('实际内存使用测试', fontweight='bold')
        ax4.set_xlabel('数据集大小')
        ax4.set_ylabel('内存使用 (MB)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_big_o_visualization(self):
        """创建Big O符号可视化"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        n = np.linspace(1, 100, 1000)
        
        # 不同复杂度函数
        complexities = {
            'O(1)': np.ones_like(n),
            'O(log n)': np.log2(n),
            'O(n)': n,
            'O(n log n)': n * np.log2(n),
            'O(n²)': n**2,
            'O(n³)': n**3,
            'O(2ⁿ)': 2**np.minimum(n/10, 20)  # 限制指数增长以便可视化
        }
        
        colors = sns.color_palette("husl", len(complexities))
        
        for i, (name, values) in enumerate(complexities.items()):
            ax.plot(n, values, label=name, linewidth=3, color=colors[i])
        
        ax.set_title('算法复杂度Big O符号对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('输入规模 (n)', fontsize=12)
        ax.set_ylabel('操作次数', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加说明文本
        ax.text(0.02, 0.98, 
                '复杂度排序（从好到坏）:\n' +
                'O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ)',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

# 示例算法复杂度数据
algorithm_complexity_data = {
    'time_complexity': {
        '线性搜索': lambda n: n,
        '二分搜索': lambda n: np.log2(n),
        '快速排序': lambda n: n * np.log2(n),
        '冒泡排序': lambda n: n**2,
        '矩阵乘法': lambda n: n**3
    },
    'space_complexity': {
        '线性搜索': lambda n: 1,
        '二分搜索': lambda n: 1,
        '快速排序': lambda n: np.log2(n),
        '冒泡排序': lambda n: 1,
        '矩阵乘法': lambda n: n**2
    },
    'actual_runtime': {
        '线性搜索': [0.001, 0.005, 0.01, 0.05, 0.1],
        '二分搜索': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        '快速排序': [0.01, 0.08, 0.2, 1.5, 3.2],
        '冒泡排序': [0.05, 1.2, 5.1, 125, 520],
        '矩阵乘法': [0.1, 15.6, 125, 15625, 125000]
    },
    'actual_memory': {
        '线性搜索': [1, 1, 1, 1, 1],
        '二分搜索': [1, 1, 1, 1, 1],
        '快速排序': [2, 4, 6, 12, 16],
        '冒泡排序': [1, 1, 1, 1, 1],
        '矩阵乘法': [10, 250, 1000, 25000, 100000]
    }
}

# 使用示例
complexity_viz = AlgorithmComplexityVisualizer()
fig7 = complexity_viz.create_complexity_comparison(algorithm_complexity_data)
fig8 = complexity_viz.create_big_o_visualization()
```

## 🎨 可视化最佳实践

### 5. 性能基准测试报告生成器

```python
class BenchmarkReportGenerator:
    """性能基准测试报告生成器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置报告样式"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set1")
    
    def generate_comprehensive_report(self, benchmark_data):
        """生成综合性能报告"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
        
        # 报告标题
        fig.suptitle('AI模型性能基准测试综合报告', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. 模型准确率对比 (第一行)
        ax1 = fig.add_subplot(gs[0, :])
        models = list(benchmark_data['accuracy'].keys())
        accuracies = list(benchmark_data['accuracy'].values())
        colors = sns.color_palette("viridis", len(models))
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. 训练时间对比 (第二行左)
        ax2 = fig.add_subplot(gs[1, 0])
        training_times = list(benchmark_data['training_time'].values())
        ax2.barh(models, training_times, color=colors, alpha=0.8)
        ax2.set_title('训练时间对比', fontweight='bold')
        ax2.set_xlabel('训练时间 (小时)')
        
        # 3. 推理速度对比 (第二行中)
        ax3 = fig.add_subplot(gs[1, 1])
        inference_speeds = list(benchmark_data['inference_speed'].values())
        ax3.barh(models, inference_speeds, color=colors, alpha=0.8)
        ax3.set_title('推理速度对比', fontweight='bold')
        ax3.set_xlabel('推理速度 (samples/sec)')
        
        # 4. 模型大小对比 (第二行右)
        ax4 = fig.add_subplot(gs[1, 2])
        model_sizes = list(benchmark_data['model_size'].values())
        ax4.barh(models, model_sizes, color=colors, alpha=0.8)
        ax4.set_title('模型大小对比', fontweight='bold')
        ax4.set_xlabel('模型大小 (MB)')
        
        # 5. 资源使用热力图 (第三行)
        ax5 = fig.add_subplot(gs[2, :])
        resource_data = np.array([
            [benchmark_data['cpu_usage'][model] for model in models],
            [benchmark_data['memory_usage'][model] for model in models],
            [benchmark_data['gpu_usage'][model] for model in models]
        ])
        
        im = ax5.imshow(resource_data, cmap='YlOrRd', aspect='auto')
        ax5.set_title('资源使用情况热力图', fontweight='bold', pad=20)
        ax5.set_xticks(range(len(models)))
        ax5.set_xticklabels(models, rotation=45, ha='right')
        ax5.set_yticks(range(3))
        ax5.set_yticklabels(['CPU使用率', '内存使用率', 'GPU使用率'])
        
        # 添加数值标注
        for i in range(3):
            for j in range(len(models)):
                text = ax5.text(j, i, f'{resource_data[i, j]:.1f}%',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # 6. 成本效益分析 (第四行左)
        ax6 = fig.add_subplot(gs[3, 0])
        costs = list(benchmark_data['cost_per_hour'].values())
        cost_efficiency = [accuracies[i] / costs[i] for i in range(len(models))]
        
        ax6.scatter(costs, accuracies, s=[size/10 for size in model_sizes], 
                   c=colors, alpha=0.7)
        ax6.set_title('成本效益分析', fontweight='bold')
        ax6.set_xlabel('每小时成本 ($)')
        ax6.set_ylabel('准确率 (%)')
        
        for i, model in enumerate(models):
            ax6.annotate(model, (costs[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 7. 扩展性分析 (第四行中)
        ax7 = fig.add_subplot(gs[3, 1])
        batch_sizes = benchmark_data['scalability']['batch_sizes']
        throughputs = benchmark_data['scalability']['throughputs']
        
        ax7.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=6)
        ax7.set_title('批处理扩展性', fontweight='bold')
        ax7.set_xlabel('批处理大小')
        ax7.set_ylabel('吞吐量 (samples/sec)')
        ax7.grid(True, alpha=0.3)
        
        # 8. 错误分析 (第四行右)
        ax8 = fig.add_subplot(gs[3, 2])
        error_types = ['分类错误', '回归误差', '过拟合', '欠拟合']
        error_counts = benchmark_data['error_analysis']
        
        wedges, texts, autotexts = ax8.pie(error_counts, labels=error_types, 
                                          autopct='%1.1f%%', startangle=90)
        ax8.set_title('错误类型分布', fontweight='bold')
        
        # 9. 性能趋势 (第五行)
        ax9 = fig.add_subplot(gs[4, :])
        dates = benchmark_data['performance_trend']['dates']
        performance_scores = benchmark_data['performance_trend']['scores']
        
        ax9.plot(dates, performance_scores, 'b-', linewidth=3, marker='o', markersize=8)
        ax9.set_title('性能改进趋势', fontweight='bold', pad=20)
        ax9.set_xlabel('日期')
        ax9.set_ylabel('综合性能评分')
        ax9.grid(True, alpha=0.3)
        
        # 旋转日期标签
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
        
        # 10. 总结表格 (第六行)
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis('tight')
        ax10.axis('off')
        
        # 创建总结表格
        summary_data = []
        for i, model in enumerate(models):
            summary_data.append([
                model,
                f"{accuracies[i]:.1f}%",
                f"{training_times[i]:.1f}h",
                f"{inference_speeds[i]}",
                f"{model_sizes[i]}MB",
                f"${costs[i]:.2f}/h"
            ])
        
        table = ax10.table(cellText=summary_data,
                          colLabels=['模型', '准确率', '训练时间', '推理速度', '模型大小', '成本'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 设置表格样式
        for i in range(len(models) + 1):
            for j in range(6):
                if i == 0:  # 表头
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax10.set_title('性能基准测试总结表', fontweight='bold', pad=20)
        
        return fig
    
    def save_report(self, fig, filename='benchmark_report.png'):
        """保存报告"""
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"报告已保存为: {filename}")

# 示例基准测试数据
benchmark_test_data = {
    'accuracy': {
        'BERT-Base': 88.5,
        'RoBERTa': 90.2,
        'DistilBERT': 86.1,
        'ALBERT': 89.3,
        'GPT-2': 87.8
    },
    'training_time': {
        'BERT-Base': 12.5,
        'RoBERTa': 15.8,
        'DistilBERT': 6.2,
        'ALBERT': 8.9,
        'GPT-2': 18.3
    },
    'inference_speed': {
        'BERT-Base': 156,
        'RoBERTa': 142,
        'DistilBERT': 312,
        'ALBERT': 198,
        'GPT-2': 89
    },
    'model_size': {
        'BERT-Base': 440,
        'RoBERTa': 498,
        'DistilBERT': 255,
        'ALBERT': 89,
        'GPT-2': 774
    },
    'cpu_usage': {
        'BERT-Base': 75.2,
        'RoBERTa': 78.5,
        'DistilBERT': 45.8,
        'ALBERT': 52.3,
        'GPT-2': 82.1
    },
    'memory_usage': {
        'BERT-Base': 68.5,
        'RoBERTa': 72.3,
        'DistilBERT': 38.9,
        'ALBERT': 25.6,
        'GPT-2': 85.4
    },
    'gpu_usage': {
        'BERT-Base': 89.2,
        'RoBERTa': 91.5,
        'DistilBERT': 65.8,
        'ALBERT': 72.1,
        'GPT-2': 94.3
    },
    'cost_per_hour': {
        'BERT-Base': 2.50,
        'RoBERTa': 3.20,
        'DistilBERT': 1.80,
        'ALBERT': 2.10,
        'GPT-2': 4.50
    },
    'scalability': {
        'batch_sizes': [1, 8, 16, 32, 64, 128],
        'throughputs': [45, 320, 580, 980, 1520, 2100]
    },
    'error_analysis': [25, 35, 20, 20],  # 对应error_types的数量
    'performance_trend': {
        'dates': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'],
        'scores': [75, 78, 82, 85, 88, 91]
    }
}

# 使用示例
report_generator = BenchmarkReportGenerator()
fig9 = report_generator.generate_comprehensive_report(benchmark_test_data)
# report_generator.save_report(fig9, 'ai_model_benchmark_report.png')
```

## 📈 总结

本文档提供了全面的性能基准测试可视化方案，包括：

### 核心功能
1. **模型性能对比** - 多维度模型评估和比较
2. **训练过程监控** - 实时训练状态可视化
3. **硬件性能分析** - GPU和系统资源评估
4. **算法复杂度分析** - 理论和实际性能对比
5. **综合报告生成** - 专业的基准测试报告

### 技术特点
- 🎨 **多样化图表** - 条形图、雷达图、热力图、趋势图
- 🔄 **交互式界面** - 支持Plotly和Streamlit交互
- 📊 **数据驱动** - 基于真实性能数据的可视化
- 🎯 **专业报告** - 适合技术文档和商业展示

### 使用建议
1. **选择合适的可视化类型** - 根据数据特点选择最佳图表
2. **注重数据质量** - 确保基准测试数据的准确性
3. **定期更新** - 保持性能数据的时效性
4. **多维度分析** - 综合考虑准确率、速度、成本等因素

这些可视化工具将帮助开发者更好地理解和优化AI模型的性能表现。