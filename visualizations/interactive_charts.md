# 交互式图表和动态可视化

本文档提供了各种交互式图表和动态可视化的实现方案，帮助读者创建更加生动和用户友好的数据展示界面。

## 🎯 交互式仪表板

### 1. 实时监控仪表板

```python
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import queue

class RealTimeMonitoringDashboard:
    """实时监控仪表板"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data_queue = queue.Queue()
        self.setup_layout()
        self.setup_callbacks()
        self.start_data_generator()
    
    def setup_layout(self):
        """设置仪表板布局"""
        self.app.layout = html.Div([
            html.H1("AI系统实时监控仪表板", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # 关键指标卡片
            html.Div([
                html.Div([
                    html.H3("QPS", style={'textAlign': 'center'}),
                    html.H2(id="qps-value", children="0", 
                           style={'textAlign': 'center', 'color': '#1f77b4'})
                ], className="metric-card", style={
                    'width': '23%', 'display': 'inline-block', 
                    'margin': '1%', 'padding': '20px',
                    'border': '1px solid #ddd', 'borderRadius': '5px'
                }),
                
                html.Div([
                    html.H3("延迟", style={'textAlign': 'center'}),
                    html.H2(id="latency-value", children="0ms", 
                           style={'textAlign': 'center', 'color': '#ff7f0e'})
                ], className="metric-card", style={
                    'width': '23%', 'display': 'inline-block', 
                    'margin': '1%', 'padding': '20px',
                    'border': '1px solid #ddd', 'borderRadius': '5px'
                }),
                
                html.Div([
                    html.H3("错误率", style={'textAlign': 'center'}),
                    html.H2(id="error-rate-value", children="0%", 
                           style={'textAlign': 'center', 'color': '#d62728'})
                ], className="metric-card", style={
                    'width': '23%', 'display': 'inline-block', 
                    'margin': '1%', 'padding': '20px',
                    'border': '1px solid #ddd', 'borderRadius': '5px'
                }),
                
                html.Div([
                    html.H3("CPU使用率", style={'textAlign': 'center'}),
                    html.H2(id="cpu-value", children="0%", 
                           style={'textAlign': 'center', 'color': '#2ca02c'})
                ], className="metric-card", style={
                    'width': '23%', 'display': 'inline-block', 
                    'margin': '1%', 'padding': '20px',
                    'border': '1px solid #ddd', 'borderRadius': '5px'
                })
            ], style={'marginBottom': 30}),
            
            # 时间序列图表
            html.Div([
                html.Div([
                    dcc.Graph(id="qps-chart")
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id="latency-chart")
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="error-chart")
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id="resource-chart")
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # 自动刷新组件
            dcc.Interval(
                id='interval-component',
                interval=1000,  # 每秒更新
                n_intervals=0
            ),
            
            # 数据存储
            dcc.Store(id='metrics-store', data={
                'timestamps': [],
                'qps': [],
                'latency': [],
                'error_rate': [],
                'cpu_usage': [],
                'memory_usage': []
            })
        ])
    
    def setup_callbacks(self):
        """设置回调函数"""
        @self.app.callback(
            [Output('metrics-store', 'data'),
             Output('qps-value', 'children'),
             Output('latency-value', 'children'),
             Output('error-rate-value', 'children'),
             Output('cpu-value', 'children')],
            [Input('interval-component', 'n_intervals')],
            [dash.dependencies.State('metrics-store', 'data')]
        )
        def update_metrics(n, stored_data):
            # 生成新的指标数据
            current_time = datetime.now()
            qps = np.random.poisson(100) + 50
            latency = np.random.exponential(50) + 10
            error_rate = np.random.beta(1, 20) * 100
            cpu_usage = np.random.beta(2, 5) * 100
            memory_usage = np.random.beta(3, 4) * 100
            
            # 更新存储的数据
            stored_data['timestamps'].append(current_time)
            stored_data['qps'].append(qps)
            stored_data['latency'].append(latency)
            stored_data['error_rate'].append(error_rate)
            stored_data['cpu_usage'].append(cpu_usage)
            stored_data['memory_usage'].append(memory_usage)
            
            # 保持最近100个数据点
            if len(stored_data['timestamps']) > 100:
                for key in stored_data:
                    stored_data[key] = stored_data[key][-100:]
            
            return (stored_data, 
                   f"{qps:.0f}",
                   f"{latency:.1f}ms",
                   f"{error_rate:.2f}%",
                   f"{cpu_usage:.1f}%")
        
        @self.app.callback(
            Output('qps-chart', 'figure'),
            [Input('metrics-store', 'data')]
        )
        def update_qps_chart(data):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['qps'],
                mode='lines+markers',
                name='QPS',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title='每秒请求数 (QPS)',
                xaxis_title='时间',
                yaxis_title='QPS',
                height=300,
                showlegend=False
            )
            return fig
        
        @self.app.callback(
            Output('latency-chart', 'figure'),
            [Input('metrics-store', 'data')]
        )
        def update_latency_chart(data):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['latency'],
                mode='lines+markers',
                name='延迟',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.update_layout(
                title='响应延迟',
                xaxis_title='时间',
                yaxis_title='延迟 (ms)',
                height=300,
                showlegend=False
            )
            return fig
        
        @self.app.callback(
            Output('error-chart', 'figure'),
            [Input('metrics-store', 'data')]
        )
        def update_error_chart(data):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['error_rate'],
                mode='lines+markers',
                name='错误率',
                line=dict(color='#d62728', width=2),
                fill='tonexty'
            ))
            fig.update_layout(
                title='错误率',
                xaxis_title='时间',
                yaxis_title='错误率 (%)',
                height=300,
                showlegend=False
            )
            return fig
        
        @self.app.callback(
            Output('resource-chart', 'figure'),
            [Input('metrics-store', 'data')]
        )
        def update_resource_chart(data):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['cpu_usage'],
                mode='lines+markers',
                name='CPU使用率',
                line=dict(color='#2ca02c', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['memory_usage'],
                mode='lines+markers',
                name='内存使用率',
                line=dict(color='#9467bd', width=2)
            ))
            fig.update_layout(
                title='资源使用率',
                xaxis_title='时间',
                yaxis_title='使用率 (%)',
                height=300,
                legend=dict(x=0, y=1)
            )
            return fig
    
    def start_data_generator(self):
        """启动数据生成器"""
        def generate_data():
            while True:
                # 模拟实时数据生成
                data = {
                    'timestamp': datetime.now(),
                    'qps': np.random.poisson(100) + 50,
                    'latency': np.random.exponential(50) + 10,
                    'error_rate': np.random.beta(1, 20) * 100,
                    'cpu_usage': np.random.beta(2, 5) * 100
                }
                self.data_queue.put(data)
                time.sleep(1)
        
        thread = threading.Thread(target=generate_data, daemon=True)
        thread.start()
    
    def run(self, debug=True, port=8050):
        """运行仪表板"""
        self.app.run_server(debug=debug, port=port)

# 使用示例
if __name__ == "__main__":
    dashboard = RealTimeMonitoringDashboard()
    dashboard.run()
```

### 2. 模型性能分析仪表板

```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class ModelPerformanceDashboard:
    """模型性能分析仪表板"""
    
    def __init__(self):
        st.set_page_config(
            page_title="模型性能分析仪表板",
            page_icon="📊",
            layout="wide"
        )
        self.setup_sidebar()
        self.main_dashboard()
    
    def setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.title("模型配置")
        
        # 数据集参数
        st.sidebar.subheader("数据集参数")
        self.n_samples = st.sidebar.slider("样本数量", 100, 5000, 1000)
        self.n_features = st.sidebar.slider("特征数量", 5, 50, 20)
        self.n_informative = st.sidebar.slider("有效特征数", 2, self.n_features, 10)
        self.test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.3)
        
        # 模型选择
        st.sidebar.subheader("模型选择")
        self.selected_models = st.sidebar.multiselect(
            "选择要比较的模型",
            ["随机森林", "逻辑回归", "支持向量机"],
            default=["随机森林", "逻辑回归"]
        )
        
        # 可视化选项
        st.sidebar.subheader("可视化选项")
        self.show_confusion_matrix = st.sidebar.checkbox("显示混淆矩阵", True)
        self.show_roc_curve = st.sidebar.checkbox("显示ROC曲线", True)
        self.show_feature_importance = st.sidebar.checkbox("显示特征重要性", True)
        self.show_learning_curve = st.sidebar.checkbox("显示学习曲线", False)
    
    def generate_data(self):
        """生成示例数据"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_features - self.n_informative,
            n_clusters_per_class=1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """训练模型"""
        models = {}
        results = {}
        
        model_configs = {
            "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
            "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
            "支持向量机": SVC(probability=True, random_state=42)
        }
        
        for model_name in self.selected_models:
            if model_name in model_configs:
                model = model_configs[model_name]
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                models[model_name] = model
                results[model_name] = {
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'accuracy': (y_pred == y_test).mean(),
                    'precision': ((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0,
                    'recall': ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
                }
                
                # 计算F1分数
                precision = results[model_name]['precision']
                recall = results[model_name]['recall']
                results[model_name]['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return models, results, y_test
    
    def create_performance_metrics_chart(self, results):
        """创建性能指标对比图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']
        
        fig = go.Figure()
        
        for model_name in results.keys():
            values = [results[model_name][metric] for metric in metrics]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="模型性能指标对比",
            showlegend=True
        )
        
        return fig
    
    def create_confusion_matrix_chart(self, y_test, results):
        """创建混淆矩阵图"""
        n_models = len(results)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(results.keys()),
            specs=[[{"type": "heatmap"}] * n_models]
        )
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['y_pred'])
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['预测负类', '预测正类'],
                    y=['实际负类', '实际正类'],
                    colorscale='Blues',
                    showscale=i == 0,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="混淆矩阵对比",
            height=400
        )
        
        return fig
    
    def create_roc_curve_chart(self, y_test, results):
        """创建ROC曲线图"""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, result) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # 添加随机分类器线
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='随机分类器',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC曲线对比',
            xaxis_title='假正率 (FPR)',
            yaxis_title='真正率 (TPR)',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def create_feature_importance_chart(self, models, X_train):
        """创建特征重要性图"""
        fig = go.Figure()
        
        feature_names = [f'特征_{i}' for i in range(X_train.shape[1])]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, model) in enumerate(models.items()):
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # 选择top 10特征
                top_indices = np.argsort(importance)[::-1][:10]
                top_features = [feature_names[idx] for idx in top_indices]
                top_importance = importance[top_indices]
                
                fig.add_trace(go.Bar(
                    x=top_features,
                    y=top_importance,
                    name=model_name,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7
                ))
        
        fig.update_layout(
            title='特征重要性对比 (Top 10)',
            xaxis_title='特征',
            yaxis_title='重要性分数',
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    def main_dashboard(self):
        """主仪表板"""
        st.title("🤖 模型性能分析仪表板")
        st.markdown("---")
        
        # 生成数据和训练模型
        if st.button("🚀 开始分析", type="primary"):
            with st.spinner("正在生成数据和训练模型..."):
                X_train, X_test, y_train, y_test = self.generate_data()
                models, results, y_test = self.train_models(X_train, X_test, y_train, y_test)
            
            if results:
                # 性能指标概览
                st.subheader("📊 性能指标概览")
                
                # 创建指标表格
                metrics_df = pd.DataFrame({
                    model_name: {
                        '准确率': f"{result['accuracy']:.3f}",
                        '精确率': f"{result['precision']:.3f}",
                        '召回率': f"{result['recall']:.3f}",
                        'F1分数': f"{result['f1']:.3f}"
                    }
                    for model_name, result in results.items()
                }).T
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # 性能指标雷达图
                col1, col2 = st.columns(2)
                
                with col1:
                    radar_fig = self.create_performance_metrics_chart(results)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    # 性能指标条形图
                    metrics_data = []
                    for model_name, result in results.items():
                        for metric in ['accuracy', 'precision', 'recall', 'f1']:
                            metrics_data.append({
                                '模型': model_name,
                                '指标': metric,
                                '数值': result[metric]
                            })
                    
                    metrics_df_long = pd.DataFrame(metrics_data)
                    bar_fig = px.bar(
                        metrics_df_long, 
                        x='指标', 
                        y='数值', 
                        color='模型',
                        title='性能指标对比',
                        barmode='group'
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # 混淆矩阵
                if self.show_confusion_matrix:
                    st.subheader("🎯 混淆矩阵分析")
                    cm_fig = self.create_confusion_matrix_chart(y_test, results)
                    st.plotly_chart(cm_fig, use_container_width=True)
                
                # ROC曲线
                if self.show_roc_curve:
                    st.subheader("📈 ROC曲线分析")
                    roc_fig = self.create_roc_curve_chart(y_test, results)
                    st.plotly_chart(roc_fig, use_container_width=True)
                
                # 特征重要性
                if self.show_feature_importance and models:
                    st.subheader("🔍 特征重要性分析")
                    fi_fig = self.create_feature_importance_chart(models, X_train)
                    st.plotly_chart(fi_fig, use_container_width=True)
                
                # 模型详细信息
                st.subheader("📋 模型详细信息")
                
                for model_name, model in models.items():
                    with st.expander(f"{model_name} 详细信息"):
                        st.write(f"**模型类型**: {type(model).__name__}")
                        st.write(f"**参数数量**: {len(str(model.get_params()))}")
                        st.write(f"**训练样本数**: {len(X_train)}")
                        st.write(f"**测试样本数**: {len(X_test)}")
                        
                        if hasattr(model, 'feature_importances_'):
                            st.write(f"**最重要特征**: 特征_{np.argmax(model.feature_importances_)}")
            else:
                st.warning("请至少选择一个模型进行分析")
        
        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 如何使用这个仪表板：
            
            1. **配置参数**: 在左侧边栏调整数据集参数和模型选择
            2. **选择模型**: 可以同时比较多个模型的性能
            3. **开始分析**: 点击"开始分析"按钮生成数据并训练模型
            4. **查看结果**: 分析各种性能指标和可视化图表
            
            ### 图表说明：
            
            - **雷达图**: 直观比较多个模型的综合性能
            - **混淆矩阵**: 分析模型的分类准确性
            - **ROC曲线**: 评估模型的分类能力
            - **特征重要性**: 了解哪些特征对预测最重要
            """)

# 使用示例
if __name__ == "__main__":
    dashboard = ModelPerformanceDashboard()
```

### 3. 数据探索分析工具

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class DataExplorationTool:
    """数据探索分析工具"""
    
    def __init__(self):
        st.set_page_config(
            page_title="数据探索分析工具",
            page_icon="🔍",
            layout="wide"
        )
        self.setup_interface()
    
    def setup_interface(self):
        """设置用户界面"""
        st.title("🔍 数据探索分析工具")
        st.markdown("上传您的数据集，进行全面的探索性数据分析")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="上传您要分析的CSV数据文件"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                self.analyze_data(df)
            except Exception as e:
                st.error(f"文件读取错误: {str(e)}")
        else:
            # 使用示例数据
            if st.button("使用示例数据"):
                df = self.generate_sample_data()
                self.analyze_data(df)
    
    def generate_sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            '年龄': np.random.normal(35, 10, n_samples).astype(int),
            '收入': np.random.lognormal(10, 0.5, n_samples),
            '教育年限': np.random.choice([12, 14, 16, 18, 20], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
            '性别': np.random.choice(['男', '女'], n_samples),
            '城市': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n_samples),
            '满意度': np.random.randint(1, 6, n_samples),
            '购买金额': np.random.exponential(500, n_samples)
        }
        
        # 添加一些相关性
        data['收入'] = data['收入'] + data['教育年限'] * 1000 + np.random.normal(0, 5000, n_samples)
        data['购买金额'] = data['购买金额'] + data['收入'] * 0.01 + np.random.normal(0, 100, n_samples)
        
        return pd.DataFrame(data)
    
    def analyze_data(self, df):
        """分析数据"""
        # 数据概览
        st.subheader("📊 数据概览")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("总列数", len(df.columns))
        with col3:
            st.metric("数值列数", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("缺失值数", df.isnull().sum().sum())
        
        # 数据预览
        st.subheader("👀 数据预览")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 数据类型和统计信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 数据类型")
            dtype_df = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.astype(str),
                '非空值数': df.count(),
                '缺失值数': df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 描述性统计")
            st.dataframe(df.describe(), use_container_width=True)
        
        # 数值列分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.analyze_numeric_columns(df, numeric_cols)
        
        # 分类列分析
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            self.analyze_categorical_columns(df, categorical_cols)
        
        # 相关性分析
        if len(numeric_cols) > 1:
            self.analyze_correlations(df, numeric_cols)
        
        # 缺失值分析
        if df.isnull().sum().sum() > 0:
            self.analyze_missing_values(df)
        
        # 异常值检测
        if numeric_cols:
            self.detect_outliers(df, numeric_cols)
    
    def analyze_numeric_columns(self, df, numeric_cols):
        """分析数值列"""
        st.subheader("🔢 数值列分析")
        
        # 选择要分析的列
        selected_numeric = st.multiselect(
            "选择要分析的数值列",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        if selected_numeric:
            # 分布图
            st.write("**分布图**")
            n_cols = min(3, len(selected_numeric))
            n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_numeric
            )
            
            for i, col in enumerate(selected_numeric):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col,
                        showlegend=False,
                        nbinsx=30
                    ),
                    row=row, col=col_idx
                )
            
            fig.update_layout(
                height=300 * n_rows,
                title="数值列分布图"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 箱线图
            st.write("**箱线图**")
            box_fig = go.Figure()
            
            for col in selected_numeric:
                box_fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers'
                ))
            
            box_fig.update_layout(
                title="数值列箱线图",
                yaxis_title="数值"
            )
            st.plotly_chart(box_fig, use_container_width=True)
    
    def analyze_categorical_columns(self, df, categorical_cols):
        """分析分类列"""
        st.subheader("📊 分类列分析")
        
        # 选择要分析的列
        selected_categorical = st.selectbox(
            "选择要分析的分类列",
            categorical_cols
        )
        
        if selected_categorical:
            col1, col2 = st.columns(2)
            
            with col1:
                # 值计数
                value_counts = df[selected_categorical].value_counts()
                st.write(f"**{selected_categorical} 值分布**")
                st.dataframe(value_counts.to_frame('计数'), use_container_width=True)
            
            with col2:
                # 饼图
                pie_fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"{selected_categorical} 分布饼图"
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            
            # 条形图
            bar_fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"{selected_categorical} 分布条形图",
                labels={'x': selected_categorical, 'y': '计数'}
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    
    def analyze_correlations(self, df, numeric_cols):
        """分析相关性"""
        st.subheader("🔗 相关性分析")
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 相关性热力图
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="相关性矩阵热力图",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 强相关性对
        st.write("**强相关性对 (|r| > 0.5)**")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        '变量1': corr_matrix.columns[i],
                        '变量2': corr_matrix.columns[j],
                        '相关系数': round(corr_val, 3)
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("没有发现强相关性对")
    
    def analyze_missing_values(self, df):
        """分析缺失值"""
        st.subheader("❓ 缺失值分析")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # 缺失值统计
                missing_df = pd.DataFrame({
                    '列名': missing_data.index,
                    '缺失值数': missing_data.values,
                    '缺失率': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            
            with col2:
                # 缺失值条形图
                bar_fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="缺失值分布",
                    labels={'x': '缺失值数量', 'y': '列名'}
                )
                st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.success("数据集中没有缺失值！")
    
    def detect_outliers(self, df, numeric_cols):
        """检测异常值"""
        st.subheader("🎯 异常值检测")
        
        # 选择检测方法
        method = st.selectbox(
            "选择异常值检测方法",
            ["IQR方法", "Z-Score方法", "改进Z-Score方法"]
        )
        
        # 选择要检测的列
        selected_col = st.selectbox(
            "选择要检测异常值的列",
            numeric_cols
        )
        
        if selected_col:
            data = df[selected_col].dropna()
            
            if method == "IQR方法":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == "Z-Score方法":
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
                
            else:  # 改进Z-Score方法
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = data[np.abs(modified_z_scores) > 3.5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("异常值数量", len(outliers))
                st.metric("异常值比例", f"{len(outliers)/len(data)*100:.2f}%")
                
                if len(outliers) > 0:
                    st.write("**异常值列表**")
                    st.dataframe(outliers.to_frame('异常值'), use_container_width=True)
            
            with col2:
                # 异常值可视化
                fig = go.Figure()
                
                # 正常值
                normal_data = data[~data.isin(outliers)]
                fig.add_trace(go.Scatter(
                    x=range(len(normal_data)),
                    y=normal_data,
                    mode='markers',
                    name='正常值',
                    marker=dict(color='blue', size=4)
                ))
                
                # 异常值
                if len(outliers) > 0:
                    outlier_indices = [i for i, val in enumerate(data) if val in outliers.values]
                    fig.add_trace(go.Scatter(
                        x=outlier_indices,
                        y=outliers,
                        mode='markers',
                        name='异常值',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig.update_layout(
                    title=f"{selected_col} 异常值检测结果",
                    xaxis_title="数据点索引",
                    yaxis_title="数值"
                )
                st.plotly_chart(fig, use_container_width=True)

# 使用示例
if __name__ == "__main__":
    tool = DataExplorationTool()
```

## 🎨 动态可视化组件

### 1. 实时数据流可视化

```python
import asyncio
import websockets
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import threading
import queue

class RealTimeDataStreamer:
    """实时数据流可视化"""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data_buffer = {
            'timestamps': [],
            'values': [],
            'categories': []
        }
        self.clients = set()
        self.running = False
    
    async def register_client(self, websocket, path):
        """注册WebSocket客户端"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def broadcast_data(self, data):
        """广播数据到所有客户端"""
        if self.clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    def generate_data_point(self):
        """生成数据点"""
        timestamp = datetime.now()
        value = np.random.normal(100, 15) + 10 * np.sin(timestamp.timestamp() / 10)
        category = np.random.choice(['A', 'B', 'C'], p=[0.5, 0.3, 0.2])
        
        return {
            'timestamp': timestamp.isoformat(),
            'value': float(value),
            'category': category
        }
    
    async def data_generator(self):
        """数据生成器"""
        while self.running:
            data_point = self.generate_data_point()
            
            # 更新缓冲区
            self.data_buffer['timestamps'].append(data_point['timestamp'])
            self.data_buffer['values'].append(data_point['value'])
            self.data_buffer['categories'].append(data_point['category'])
            
            # 保持缓冲区大小
            if len(self.data_buffer['timestamps']) > self.max_points:
                for key in self.data_buffer:
                    self.data_buffer[key] = self.data_buffer[key][-self.max_points:]
            
            # 广播数据
            await self.broadcast_data({
                'type': 'data_point',
                'data': data_point,
                'buffer': self.data_buffer
            })
            
            await asyncio.sleep(0.1)  # 100ms间隔
    
    def start_server(self, host='localhost', port=8765):
        """启动WebSocket服务器"""
        self.running = True
        
        async def main():
            # 启动WebSocket服务器
            server = await websockets.serve(
                self.register_client, host, port
            )
            
            # 启动数据生成器
            data_task = asyncio.create_task(self.data_generator())
            
            print(f"实时数据流服务器启动: ws://{host}:{port}")
            
            try:
                await server.wait_closed()
            except KeyboardInterrupt:
                self.running = False
                data_task.cancel()
                server.close()
        
        asyncio.run(main())

# 客户端HTML页面
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>实时数据流可视化</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .chart { margin: 20px 0; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .connected { background-color: #d4edda; color: #155724; }
        .disconnected { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 实时数据流可视化</h1>
        <div id="status" class="status disconnected">连接中...</div>
        
        <div class="chart">
            <div id="timeseries-chart"></div>
        </div>
        
        <div class="chart">
            <div id="histogram-chart"></div>
        </div>
        
        <div class="chart">
            <div id="category-chart"></div>
        </div>
    </div>

    <script>
        // WebSocket连接
        const ws = new WebSocket('ws://localhost:8765');
        const statusDiv = document.getElementById('status');
        
        // 数据缓冲区
        let dataBuffer = {
            timestamps: [],
            values: [],
            categories: []
        };
        
        // 初始化图表
        function initCharts() {
            // 时间序列图
            Plotly.newPlot('timeseries-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: '实时数据',
                line: { color: '#1f77b4', width: 2 }
            }], {
                title: '实时时间序列数据',
                xaxis: { title: '时间' },
                yaxis: { title: '数值' }
            });
            
            // 直方图
            Plotly.newPlot('histogram-chart', [{
                x: [],
                type: 'histogram',
                nbinsx: 20,
                name: '数值分布'
            }], {
                title: '数值分布直方图',
                xaxis: { title: '数值' },
                yaxis: { title: '频次' }
            });
            
            // 分类饼图
            Plotly.newPlot('category-chart', [{
                values: [],
                labels: [],
                type: 'pie',
                name: '分类分布'
            }], {
                title: '分类分布饼图'
            });
        }
        
        // 更新图表
        function updateCharts(buffer) {
            // 更新时间序列图
            Plotly.restyle('timeseries-chart', {
                x: [buffer.timestamps],
                y: [buffer.values]
            });
            
            // 更新直方图
            Plotly.restyle('histogram-chart', {
                x: [buffer.values]
            });
            
            // 更新分类饼图
            const categoryCounts = {};
            buffer.categories.forEach(cat => {
                categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
            });
            
            Plotly.restyle('category-chart', {
                values: [Object.values(categoryCounts)],
                labels: [Object.keys(categoryCounts)]
            });
        }
        
        // WebSocket事件处理
        ws.onopen = function(event) {
            statusDiv.textContent = '✅ 已连接到实时数据流';
            statusDiv.className = 'status connected';
            initCharts();
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'data_point') {
                dataBuffer = message.buffer;
                updateCharts(dataBuffer);
            }
        };
        
        ws.onclose = function(event) {
            statusDiv.textContent = '❌ 连接已断开';
            statusDiv.className = 'status disconnected';
        };
        
        ws.onerror = function(error) {
            statusDiv.textContent = '❌ 连接错误';
            statusDiv.className = 'status disconnected';
            console.error('WebSocket错误:', error);
        };
    </script>
</body>
</html>
"""

# 使用示例
if __name__ == "__main__":
    # 保存HTML文件
    with open('realtime_visualization.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    print("HTML文件已生成: realtime_visualization.html")
    print("请在浏览器中打开该文件，然后启动数据流服务器")
    
    # 启动实时数据流服务器
    streamer = RealTimeDataStreamer()
    streamer.start_server()
```

### 2. 交互式网络图

```python
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import streamlit as st

class InteractiveNetworkGraph:
    """交互式网络图"""
    
    def __init__(self):
        self.graph = None
        self.pos = None
        self.setup_interface()
    
    def setup_interface(self):
        """设置用户界面"""
        st.title("🕸️ 交互式网络图分析")
        
        # 侧边栏配置
        st.sidebar.title("网络配置")
        
        # 网络类型选择
        network_type = st.sidebar.selectbox(
            "选择网络类型",
            ["随机网络", "小世界网络", "无标度网络", "社交网络", "知识图谱"]
        )
        
        # 网络参数
        n_nodes = st.sidebar.slider("节点数量", 10, 200, 50)
        
        if network_type == "随机网络":
            p = st.sidebar.slider("连接概率", 0.01, 0.5, 0.1)
            self.graph = nx.erdos_renyi_graph(n_nodes, p)
            
        elif network_type == "小世界网络":
            k = st.sidebar.slider("邻居数量", 2, 10, 4)
            p = st.sidebar.slider("重连概率", 0.01, 1.0, 0.3)
            self.graph = nx.watts_strogatz_graph(n_nodes, k, p)
            
        elif network_type == "无标度网络":
            m = st.sidebar.slider("新节点连接数", 1, 10, 3)
            self.graph = nx.barabasi_albert_graph(n_nodes, m)
            
        elif network_type == "社交网络":
            self.graph = self.create_social_network(n_nodes)
            
        else:  # 知识图谱
            self.graph = self.create_knowledge_graph(n_nodes)
        
        # 布局选择
        layout_type = st.sidebar.selectbox(
            "选择布局算法",
            ["spring", "circular", "random", "shell", "spectral"]
        )
        
        # 计算布局
        if layout_type == "spring":
            self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout_type == "circular":
            self.pos = nx.circular_layout(self.graph)
        elif layout_type == "random":
            self.pos = nx.random_layout(self.graph)
        elif layout_type == "shell":
            self.pos = nx.shell_layout(self.graph)
        else:
            self.pos = nx.spectral_layout(self.graph)
        
        # 显示网络分析
        self.analyze_network()
        
        # 显示交互式图表
        self.create_interactive_graph()
    
    def create_social_network(self, n_nodes):
        """创建社交网络"""
        # 创建社区结构
        n_communities = max(3, n_nodes // 15)
        community_sizes = np.random.multinomial(n_nodes, [1/n_communities] * n_communities)
        
        graph = nx.Graph()
        node_id = 0
        communities = []
        
        for i, size in enumerate(community_sizes):
            community_nodes = list(range(node_id, node_id + size))
            communities.append(community_nodes)
            
            # 社区内连接（高密度）
            for j in range(len(community_nodes)):
                for k in range(j + 1, len(community_nodes)):
                    if np.random.random() < 0.3:  # 社区内连接概率
                        graph.add_edge(community_nodes[j], community_nodes[k])
            
            node_id += size
        
        # 社区间连接（低密度）
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                for node1 in communities[i]:
                    for node2 in communities