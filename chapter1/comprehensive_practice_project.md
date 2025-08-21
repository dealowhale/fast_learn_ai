# 综合实践项目：电商数据分析系统

## 项目概述

本项目将综合运用第1章所学的传统AI算法，构建一个完整的电商数据分析系统。通过真实的业务场景，让你体验从数据预处理到模型部署的完整机器学习工作流。

### 项目目标

- 掌握端到端的机器学习项目开发流程
- 综合应用多种算法解决实际业务问题
- 学会模型选择、评估和优化的方法
- 培养数据科学思维和问题解决能力

### 技术栈

- **数据处理**：Pandas, NumPy
- **机器学习**：Scikit-learn
- **可视化**：Matplotlib, Seaborn, Plotly
- **开发环境**：Trae AI IDE

## 业务背景

假设你是某电商平台的数据科学家，需要为公司构建智能化的数据分析系统，主要解决以下业务问题：

1. **客户细分**：识别不同价值的客户群体
2. **销量预测**：预测商品未来销量趋势
3. **推荐系统**：为用户推荐相关商品
4. **流失预测**：识别可能流失的客户
5. **关联分析**：发现商品之间的关联规律

## 数据集设计

### 数据生成器

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EcommerceDataGenerator:
    """电商数据生成器"""
    
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        self.random_state = random_state
        
    def generate_customers(self, n_customers=5000):
        """生成客户数据"""
        customers = []
        
        for i in range(n_customers):
            # 客户基本信息
            customer_id = f"C{i+1:06d}"
            age = np.random.normal(35, 12)
            age = max(18, min(70, age))  # 限制年龄范围
            
            # 根据年龄生成收入
            if age < 25:
                income = np.random.normal(4000, 1500)
            elif age < 35:
                income = np.random.normal(8000, 3000)
            elif age < 50:
                income = np.random.normal(12000, 4000)
            else:
                income = np.random.normal(10000, 3500)
            
            income = max(2000, income)  # 最低收入
            
            # 性别
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            
            # 城市等级
            city_tier = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            
            # 注册时间
            days_ago = np.random.randint(30, 1095)  # 1个月到3年前
            register_date = datetime.now() - timedelta(days=days_ago)
            
            customers.append({
                'customer_id': customer_id,
                'age': int(age),
                'income': round(income, 2),
                'gender': gender,
                'city_tier': city_tier,
                'register_date': register_date
            })
        
        return pd.DataFrame(customers)
    
    def generate_products(self, n_products=1000):
        """生成商品数据"""
        categories = ['电子产品', '服装', '家居', '美妆', '食品', '图书', '运动', '母婴']
        brands = ['品牌A', '品牌B', '品牌C', '品牌D', '品牌E', '其他']
        
        products = []
        
        for i in range(n_products):
            product_id = f"P{i+1:06d}"
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # 根据类别生成价格
            if category == '电子产品':
                price = np.random.lognormal(6, 1)  # 较高价格
            elif category == '服装':
                price = np.random.lognormal(4, 0.8)
            elif category == '家居':
                price = np.random.lognormal(4.5, 1.2)
            elif category == '美妆':
                price = np.random.lognormal(3.5, 0.7)
            elif category == '食品':
                price = np.random.lognormal(2.5, 0.5)
            elif category == '图书':
                price = np.random.lognormal(2.8, 0.4)
            elif category == '运动':
                price = np.random.lognormal(4.2, 0.9)
            else:  # 母婴
                price = np.random.lognormal(3.8, 0.8)
            
            price = max(10, price)  # 最低价格
            
            # 商品评分
            rating = np.random.beta(8, 2) * 5  # 偏向高评分
            rating = max(1, min(5, rating))
            
            # 库存
            stock = np.random.poisson(100)
            
            products.append({
                'product_id': product_id,
                'category': category,
                'brand': brand,
                'price': round(price, 2),
                'rating': round(rating, 1),
                'stock': stock
            })
        
        return pd.DataFrame(products)
    
    def generate_transactions(self, customers_df, products_df, n_transactions=20000):
        """生成交易数据"""
        transactions = []
        
        for i in range(n_transactions):
            # 选择客户
            customer = customers_df.sample(1).iloc[0]
            
            # 根据客户特征选择商品
            # 高收入客户更倾向于购买高价商品
            if customer['income'] > 10000:
                product_weights = products_df['price'] / products_df['price'].sum()
            else:
                product_weights = (1 / products_df['price']) / (1 / products_df['price']).sum()
            
            product = products_df.sample(1, weights=product_weights).iloc[0]
            
            # 购买数量
            if product['price'] > 500:
                quantity = np.random.poisson(1) + 1  # 高价商品买得少
            else:
                quantity = np.random.poisson(2) + 1  # 低价商品买得多
            
            quantity = min(quantity, product['stock'])  # 不能超过库存
            
            # 交易时间
            days_ago = np.random.randint(0, 365)
            transaction_date = datetime.now() - timedelta(days=days_ago)
            
            # 总金额
            total_amount = product['price'] * quantity
            
            # 支付方式
            payment_method = np.random.choice(['信用卡', '支付宝', '微信支付'], 
                                            p=[0.3, 0.4, 0.3])
            
            transactions.append({
                'transaction_id': f"T{i+1:08d}",
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'quantity': quantity,
                'unit_price': product['price'],
                'total_amount': round(total_amount, 2),
                'transaction_date': transaction_date,
                'payment_method': payment_method
            })
        
        return pd.DataFrame(transactions)
    
    def generate_customer_behavior(self, customers_df, transactions_df):
        """生成客户行为特征"""
        behavior_data = []
        
        for _, customer in customers_df.iterrows():
            customer_transactions = transactions_df[
                transactions_df['customer_id'] == customer['customer_id']
            ]
            
            if len(customer_transactions) == 0:
                # 没有交易记录的客户
                behavior_data.append({
                    'customer_id': customer['customer_id'],
                    'total_orders': 0,
                    'total_amount': 0,
                    'avg_order_value': 0,
                    'days_since_last_order': 999,
                    'favorite_category': 'None',
                    'is_active': 0
                })
                continue
            
            # 计算行为特征
            total_orders = len(customer_transactions)
            total_amount = customer_transactions['total_amount'].sum()
            avg_order_value = total_amount / total_orders
            
            # 最后一次购买距今天数
            last_order_date = customer_transactions['transaction_date'].max()
            days_since_last_order = (datetime.now() - last_order_date).days
            
            # 最喜欢的商品类别
            # 需要关联商品表
            customer_products = customer_transactions.merge(
                products_df, on='product_id', how='left'
            )
            
            if len(customer_products) > 0:
                favorite_category = customer_products['category'].mode().iloc[0]
            else:
                favorite_category = 'Unknown'
            
            # 是否活跃客户（30天内有购买）
            is_active = 1 if days_since_last_order <= 30 else 0
            
            behavior_data.append({
                'customer_id': customer['customer_id'],
                'total_orders': total_orders,
                'total_amount': round(total_amount, 2),
                'avg_order_value': round(avg_order_value, 2),
                'days_since_last_order': days_since_last_order,
                'favorite_category': favorite_category,
                'is_active': is_active
            })
        
        return pd.DataFrame(behavior_data)

# 生成完整数据集
def create_ecommerce_dataset():
    """创建完整的电商数据集"""
    generator = EcommerceDataGenerator()
    
    print("正在生成电商数据集...")
    
    # 生成各类数据
    customers_df = generator.generate_customers(5000)
    products_df = generator.generate_products(1000)
    transactions_df = generator.generate_transactions(customers_df, products_df, 20000)
    behavior_df = generator.generate_customer_behavior(customers_df, transactions_df)
    
    print(f"生成完成！")
    print(f"客户数量: {len(customers_df)}")
    print(f"商品数量: {len(products_df)}")
    print(f"交易数量: {len(transactions_df)}")
    
    return customers_df, products_df, transactions_df, behavior_df

# 数据探索分析
class EcommerceEDA:
    """电商数据探索分析"""
    
    def __init__(self, customers_df, products_df, transactions_df, behavior_df):
        self.customers_df = customers_df
        self.products_df = products_df
        self.transactions_df = transactions_df
        self.behavior_df = behavior_df
        
    def basic_statistics(self):
        """基础统计信息"""
        print("=== 数据集基础统计 ===")
        print(f"客户总数: {len(self.customers_df):,}")
        print(f"商品总数: {len(self.products_df):,}")
        print(f"交易总数: {len(self.transactions_df):,}")
        print(f"总交易金额: ¥{self.transactions_df['total_amount'].sum():,.2f}")
        print(f"平均订单价值: ¥{self.transactions_df['total_amount'].mean():.2f}")
        
        # 活跃客户统计
        active_customers = self.behavior_df['is_active'].sum()
        print(f"活跃客户数: {active_customers:,} ({active_customers/len(self.customers_df)*100:.1f}%)")
        
    def visualize_customer_distribution(self):
        """可视化客户分布"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 年龄分布
        axes[0, 0].hist(self.customers_df['age'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('客户年龄分布')
        axes[0, 0].set_xlabel('年龄')
        axes[0, 0].set_ylabel('客户数量')
        
        # 收入分布
        axes[0, 1].hist(self.customers_df['income'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('客户收入分布')
        axes[0, 1].set_xlabel('月收入')
        axes[0, 1].set_ylabel('客户数量')
        
        # 性别分布
        gender_counts = self.customers_df['gender'].value_counts()
        axes[0, 2].pie(gender_counts.values, labels=['女性', '男性'], autopct='%1.1f%%')
        axes[0, 2].set_title('客户性别分布')
        
        # 城市等级分布
        city_counts = self.customers_df['city_tier'].value_counts().sort_index()
        axes[1, 0].bar(city_counts.index, city_counts.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('客户城市等级分布')
        axes[1, 0].set_xlabel('城市等级')
        axes[1, 0].set_ylabel('客户数量')
        
        # 消费金额分布
        axes[1, 1].hist(self.behavior_df['total_amount'], bins=30, alpha=0.7, color='pink')
        axes[1, 1].set_title('客户总消费金额分布')
        axes[1, 1].set_xlabel('总消费金额')
        axes[1, 1].set_ylabel('客户数量')
        
        # 订单数量分布
        axes[1, 2].hist(self.behavior_df['total_orders'], bins=30, alpha=0.7, color='lightcoral')
        axes[1, 2].set_title('客户订单数量分布')
        axes[1, 2].set_xlabel('订单数量')
        axes[1, 2].set_ylabel('客户数量')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_product_analysis(self):
        """商品分析可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 商品类别分布
        category_counts = self.products_df['category'].value_counts()
        axes[0, 0].bar(range(len(category_counts)), category_counts.values, color='lightblue')
        axes[0, 0].set_title('商品类别分布')
        axes[0, 0].set_xlabel('商品类别')
        axes[0, 0].set_ylabel('商品数量')
        axes[0, 0].set_xticks(range(len(category_counts)))
        axes[0, 0].set_xticklabels(category_counts.index, rotation=45)
        
        # 价格分布
        axes[0, 1].hist(self.products_df['price'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('商品价格分布')
        axes[0, 1].set_xlabel('价格')
        axes[0, 1].set_ylabel('商品数量')
        
        # 评分分布
        axes[1, 0].hist(self.products_df['rating'], bins=20, alpha=0.7, color='gold')
        axes[1, 0].set_title('商品评分分布')
        axes[1, 0].set_xlabel('评分')
        axes[1, 0].set_ylabel('商品数量')
        
        # 各类别平均价格
        avg_price_by_category = self.products_df.groupby('category')['price'].mean().sort_values(ascending=False)
        axes[1, 1].bar(range(len(avg_price_by_category)), avg_price_by_category.values, color='orange')
        axes[1, 1].set_title('各类别平均价格')
        axes[1, 1].set_xlabel('商品类别')
        axes[1, 1].set_ylabel('平均价格')
        axes[1, 1].set_xticks(range(len(avg_price_by_category)))
        axes[1, 1].set_xticklabels(avg_price_by_category.index, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_sales_trends(self):
        """销售趋势分析"""
        # 按日期聚合销售数据
        self.transactions_df['date'] = self.transactions_df['transaction_date'].dt.date
        daily_sales = self.transactions_df.groupby('date').agg({
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        daily_sales.columns = ['date', 'daily_revenue', 'daily_orders']
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 每日收入趋势
        axes[0].plot(daily_sales['date'], daily_sales['daily_revenue'], color='blue', linewidth=1)
        axes[0].set_title('每日收入趋势')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('收入')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 每日订单数趋势
        axes[1].plot(daily_sales['date'], daily_sales['daily_orders'], color='red', linewidth=1)
        axes[1].set_title('每日订单数趋势')
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('订单数')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return daily_sales
```

## 任务1：客户细分

### 目标
使用K-means聚类算法对客户进行分群，识别不同价值的客户群体。

### 实现代码

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

class CustomerSegmentation:
    """客户细分分析"""
    
    def __init__(self, customers_df, behavior_df):
        self.customers_df = customers_df
        self.behavior_df = behavior_df
        self.merged_df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def prepare_features(self):
        """准备特征数据"""
        # 合并客户基本信息和行为数据
        self.merged_df = self.customers_df.merge(self.behavior_df, on='customer_id')
        
        # 选择用于聚类的特征
        features = [
            'age', 'income', 'total_orders', 'total_amount', 
            'avg_order_value', 'days_since_last_order'
        ]
        
        # 处理缺失值
        feature_data = self.merged_df[features].fillna(0)
        
        # 标准化特征
        scaled_features = self.scaler.fit_transform(feature_data)
        
        return scaled_features, features
    
    def find_optimal_clusters(self, max_clusters=10):
        """寻找最优聚类数量"""
        scaled_features, _ = self.prepare_features()
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
        
        # 可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 肘部法则
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_title('肘部法则')
        axes[0].set_xlabel('聚类数量')
        axes[0].set_ylabel('簇内平方和')
        axes[0].grid(True)
        
        # 轮廓系数
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_title('轮廓系数')
        axes[1].set_xlabel('聚类数量')
        axes[1].set_ylabel('轮廓系数')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 推荐最优聚类数
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"推荐聚类数量: {best_k} (轮廓系数: {max(silhouette_scores):.3f})")
        
        return best_k
    
    def perform_clustering(self, n_clusters=4):
        """执行客户聚类"""
        scaled_features, feature_names = self.prepare_features()
        
        # 训练K-means模型
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(scaled_features)
        
        # 添加聚类标签
        self.merged_df['cluster'] = cluster_labels
        
        # 分析各聚类特征
        cluster_summary = self.merged_df.groupby('cluster').agg({
            'age': 'mean',
            'income': 'mean',
            'total_orders': 'mean',
            'total_amount': 'mean',
            'avg_order_value': 'mean',
            'days_since_last_order': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        cluster_summary.columns = [
            '平均年龄', '平均收入', '平均订单数', '平均消费金额', 
            '平均订单价值', '平均距上次购买天数', '客户数量'
        ]
        
        print("=== 客户群体特征分析 ===")
        print(cluster_summary)
        
        return cluster_labels, cluster_summary
    
    def visualize_clusters(self):
        """可视化聚类结果"""
        if self.merged_df is None or 'cluster' not in self.merged_df.columns:
            print("请先执行聚类分析")
            return
        
        # 2D可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 收入 vs 消费金额
        scatter1 = axes[0, 0].scatter(self.merged_df['income'], 
                                     self.merged_df['total_amount'],
                                     c=self.merged_df['cluster'], 
                                     cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('收入')
        axes[0, 0].set_ylabel('总消费金额')
        axes[0, 0].set_title('收入 vs 总消费金额')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 年龄 vs 订单数
        scatter2 = axes[0, 1].scatter(self.merged_df['age'], 
                                     self.merged_df['total_orders'],
                                     c=self.merged_df['cluster'], 
                                     cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('年龄')
        axes[0, 1].set_ylabel('总订单数')
        axes[0, 1].set_title('年龄 vs 总订单数')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 平均订单价值 vs 距上次购买天数
        scatter3 = axes[1, 0].scatter(self.merged_df['avg_order_value'], 
                                     self.merged_df['days_since_last_order'],
                                     c=self.merged_df['cluster'], 
                                     cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('平均订单价值')
        axes[1, 0].set_ylabel('距上次购买天数')
        axes[1, 0].set_title('平均订单价值 vs 距上次购买天数')
        plt.colorbar(scatter3, ax=axes[1, 0])
        
        # 聚类分布
        cluster_counts = self.merged_df['cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, 
                      color=['red', 'blue', 'green', 'orange'][:len(cluster_counts)])
        axes[1, 1].set_xlabel('聚类')
        axes[1, 1].set_ylabel('客户数量')
        axes[1, 1].set_title('各聚类客户数量分布')
        
        plt.tight_layout()
        plt.show()
    
    def interpret_clusters(self):
        """解释聚类结果"""
        if self.merged_df is None or 'cluster' not in self.merged_df.columns:
            print("请先执行聚类分析")
            return
        
        cluster_interpretations = {
            0: "高价值客户：收入高、消费多、订单频繁",
            1: "潜力客户：年轻、收入中等、有增长潜力", 
            2: "流失风险客户：长时间未购买、需要挽回",
            3: "普通客户：基础消费群体、维持现状"
        }
        
        print("=== 客户群体解释 ===")
        for cluster_id in sorted(self.merged_df['cluster'].unique()):
            cluster_data = self.merged_df[self.merged_df['cluster'] == cluster_id]
            count = len(cluster_data)
            percentage = count / len(self.merged_df) * 100
            
            print(f"\n聚类 {cluster_id}: {cluster_interpretations.get(cluster_id, '待分析群体')}")
            print(f"客户数量: {count} ({percentage:.1f}%)")
            print(f"平均收入: ¥{cluster_data['income'].mean():.0f}")
            print(f"平均消费: ¥{cluster_data['total_amount'].mean():.0f}")
            print(f"平均订单数: {cluster_data['total_orders'].mean():.1f}")
```

## 任务2：销量预测

### 目标
使用线性回归预测商品未来销量趋势。

### 实现代码

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

class SalesForecast:
    """销量预测分析"""
    
    def __init__(self, transactions_df, products_df):
        self.transactions_df = transactions_df
        self.products_df = products_df
        self.sales_data = None
        self.models = {}
        
    def prepare_sales_data(self):
        """准备销量数据"""
        # 合并交易和商品数据
        merged_data = self.transactions_df.merge(self.products_df, on='product_id')
        
        # 按商品和日期聚合销量
        merged_data['date'] = merged_data['transaction_date'].dt.date
        
        sales_summary = merged_data.groupby(['product_id', 'date']).agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'price': 'first',
            'category': 'first',
            'rating': 'first'
        }).reset_index()
        
        # 添加时间特征
        sales_summary['date'] = pd.to_datetime(sales_summary['date'])
        sales_summary['day_of_week'] = sales_summary['date'].dt.dayofweek
        sales_summary['month'] = sales_summary['date'].dt.month
        sales_summary['day_of_year'] = sales_summary['date'].dt.dayofyear
        
        # 添加滞后特征（前几天的销量）
        sales_summary = sales_summary.sort_values(['product_id', 'date'])
        sales_summary['quantity_lag1'] = sales_summary.groupby('product_id')['quantity'].shift(1)
        sales_summary['quantity_lag7'] = sales_summary.groupby('product_id')['quantity'].shift(7)
        
        # 添加移动平均特征
        sales_summary['quantity_ma7'] = sales_summary.groupby('product_id')['quantity'].rolling(7).mean().reset_index(0, drop=True)
        
        self.sales_data = sales_summary.dropna()
        
        return self.sales_data
    
    def create_features(self):
        """创建预测特征"""
        if self.sales_data is None:
            self.prepare_sales_data()
        
        # 选择特征
        feature_columns = [
            'price', 'rating', 'day_of_week', 'month', 'day_of_year',
            'quantity_lag1', 'quantity_lag7', 'quantity_ma7'
        ]
        
        # 处理类别特征
        category_dummies = pd.get_dummies(self.sales_data['category'], prefix='category')
        
        # 合并特征
        X = pd.concat([
            self.sales_data[feature_columns],
            category_dummies
        ], axis=1)
        
        y = self.sales_data['quantity']
        
        return X, y
    
    def train_models(self):
        """训练预测模型"""
        X, y = self.create_features()
        
        # 划分训练集和测试集（按时间顺序）
        split_date = self.sales_data['date'].quantile(0.8)
        train_mask = self.sales_data['date'] <= split_date
        
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        
        # 训练多个模型
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'y_pred_test': y_pred_test
            }
            
            self.models[name] = model
        
        # 显示结果
        print("=== 模型性能对比 ===")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  训练集 R²: {result['train_r2']:.3f}")
            print(f"  测试集 R²: {result['test_r2']:.3f}")
            print(f"  训练集 MAE: {result['train_mae']:.3f}")
            print(f"  测试集 MAE: {result['test_mae']:.3f}")
        
        return results, X_test, y_test
    
    def visualize_predictions(self, results, X_test, y_test):
        """可视化预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 预测 vs 实际值散点图
        for i, (name, result) in enumerate(results.items()):
            row, col = i // 2, i % 2
            
            axes[row, col].scatter(y_test, result['y_pred_test'], alpha=0.6)
            axes[row, col].plot([y_test.min(), y_test.max()], 
                              [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[row, col].set_xlabel('实际销量')
            axes[row, col].set_ylabel('预测销量')
            axes[row, col].set_title(f'{name} - 预测 vs 实际')
            axes[row, col].text(0.05, 0.95, f'R² = {result["test_r2"]:.3f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 残差分析
        for i, (name, result) in enumerate(results.items()):
            if i >= 2:  # 只显示前两个模型的残差
                break
            row, col = 1, i
            
            residuals = y_test - result['y_pred_test']
            axes[row, col].scatter(result['y_pred_test'], residuals, alpha=0.6)
            axes[row, col].axhline(y=0, color='r', linestyle='--')
            axes[row, col].set_xlabel('预测销量')
            axes[row, col].set_ylabel('残差')
            axes[row, col].set_title(f'{name} - 残差分析')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        if 'Random Forest' not in self.models:
            print("请先训练随机森林模型")
            return
        
        X, _ = self.create_features()
        rf_model = self.models['Random Forest']
        
        # 获取特征重要性
        importances = rf_model.feature_importances_
        feature_names = X.columns
        
        # 排序
        indices = np.argsort(importances)[::-1]
        
        # 可视化
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title('销量预测特征重要性')
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 打印前10个重要特征
        print("=== 前10个重要特征 ===")
        for i in range(min(10, len(importances))):
            idx = indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
```

## 任务3：商品推荐系统

### 目标
使用K近邻算法实现基于用户行为的商品推荐。

### 实现代码

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class RecommendationSystem:
    """商品推荐系统"""
    
    def __init__(self, transactions_df, products_df, customers_df):
        self.transactions_df = transactions_df
        self.products_df = products_df
        self.customers_df = customers_df
        self.user_item_matrix = None
        self.item_features = None
        self.knn_model = None
        
    def create_user_item_matrix(self):
        """创建用户-商品矩阵"""
        # 计算用户对商品的评分（基于购买次数和金额）
        user_item_data = self.transactions_df.groupby(['customer_id', 'product_id']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # 创建评分（结合购买次数和金额）
        user_item_data['rating'] = (
            user_item_data['quantity'] * 0.3 + 
            user_item_data['total_amount'] / user_item_data['total_amount'].max() * 0.7
        )
        
        # 创建透视表
        self.user_item_matrix = user_item_data.pivot_table(
            index='customer_id', 
            columns='product_id', 
            values='rating', 
            fill_value=0
        )
        
        return self.user_item_matrix
    
    def prepare_item_features(self):
        """准备商品特征"""
        # 商品基础特征
        item_features = self.products_df.copy()
        
        # 类别编码
        category_dummies = pd.get_dummies(item_features['category'], prefix='category')
        brand_dummies = pd.get_dummies(item_features['brand'], prefix='brand')
        
        # 合并特征
        self.item_features = pd.concat([
            item_features[['product_id', 'price', 'rating']],
            category_dummies,
            brand_dummies
        ], axis=1)
        
        return self.item_features
    
    def train_collaborative_filtering(self):
        """训练协同过滤模型"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # 使用KNN进行协同过滤
        self.knn_model = NearestNeighbors(
            n_neighbors=20, 
            metric='cosine', 
            algorithm='brute'
        )
        
        # 训练模型（基于用户相似性）
        self.knn_model.fit(self.user_item_matrix.values)
        
        return self.knn_model
    
    def get_user_recommendations(self, customer_id, n_recommendations=10):
        """为用户推荐商品"""
        if self.knn_model is None:
            self.train_collaborative_filtering()
        
        if customer_id not in self.user_item_matrix.index:
            print(f"客户 {customer_id} 不在数据中")
            return []
        
        # 获取用户索引
        user_idx = self.user_item_matrix.index.get_loc(customer_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # 找到相似用户
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=6)
        similar_users = indices.flatten()[1:]  # 排除自己
        
        # 获取相似用户喜欢的商品
        recommendations = {}
        user_purchased = set(self.user_item_matrix.columns[
            self.user_item_matrix.iloc[user_idx] > 0
        ])
        
        for similar_user_idx in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            
            for product_id, rating in similar_user_ratings.items():
                if rating > 0 and product_id not in user_purchased:
                    if product_id not in recommendations:
                        recommendations[product_id] = 0
                    recommendations[product_id] += rating
        
        # 排序并返回推荐
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        # 添加商品信息
        recommendation_details = []
        for product_id, score in sorted_recommendations:
            product_info = self.products_df[
                self.products_df['product_id'] == product_id
            ].iloc[0]
            
            recommendation_details.append({
                'product_id': product_id,
                'category': product_info['category'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'recommendation_score': score
            })
        
        return recommendation_details
    
    def content_based_recommendations(self, customer_id, n_recommendations=10):
        """基于内容的推荐"""
        if self.item_features is None:
            self.prepare_item_features()
        
        # 获取用户历史购买记录
        user_purchases = self.transactions_df[
            self.transactions_df['customer_id'] == customer_id
        ]['product_id'].unique()
        
        if len(user_purchases) == 0:
            print(f"客户 {customer_id} 没有购买记录")
            return []
        
        # 计算用户偏好向量（基于购买商品的平均特征）
        purchased_features = self.item_features[
            self.item_features['product_id'].isin(user_purchases)
        ]
        
        # 数值特征
        numeric_cols = ['price', 'rating']
        user_profile = purchased_features[numeric_cols].mean()
        
        # 类别特征（取最常购买的类别）
        category_cols = [col for col in self.item_features.columns if col.startswith('category_')]
        brand_cols = [col for col in self.item_features.columns if col.startswith('brand_')]
        
        user_category_profile = purchased_features[category_cols].mean()
        user_brand_profile = purchased_features[brand_cols].mean()
        
        # 合并用户偏好
        user_profile_full = pd.concat([
            user_profile, user_category_profile, user_brand_profile
        ])
        
        # 计算与所有商品的相似度
        candidate_items = self.item_features[
            ~self.item_features['product_id'].isin(user_purchases)
        ]
        
        similarities = []
        for _, item in candidate_items.iterrows():
            item_features_vector = item[numeric_cols + category_cols + brand_cols]
            similarity = cosine_similarity(
                user_profile_full.values.reshape(1, -1),
                item_features_vector.values.reshape(1, -1)
            )[0][0]
            similarities.append((item['product_id'], similarity))
        
        # 排序并返回推荐
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = similarities[:n_recommendations]
        
        # 添加商品详细信息
        recommendation_details = []
        for product_id, similarity in top_recommendations:
            product_info = self.products_df[
                self.products_df['product_id'] == product_id
            ].iloc[0]
            
            recommendation_details.append({
                'product_id': product_id,
                'category': product_info['category'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'similarity_score': similarity
            })
        
        return recommendation_details
    
    def evaluate_recommendations(self, test_ratio=0.2):
        """评估推荐系统性能"""
        # 划分训练集和测试集
        users = self.user_item_matrix.index.tolist()
        test_users = np.random.choice(users, int(len(users) * test_ratio), replace=False)
        
        precision_scores = []
        recall_scores = []
        
        for user in test_users:
            # 获取用户实际购买的商品
            actual_purchases = set(self.transactions_df[
                self.transactions_df['customer_id'] == user
            ]['product_id'].unique())
            
            if len(actual_purchases) < 2:
                continue
            
            # 隐藏部分购买记录作为测试
            test_items = set(np.random.choice(
                list(actual_purchases), 
                min(3, len(actual_purchases)//2), 
                replace=False
            ))
            
            # 基于剩余购买记录进行推荐
            recommendations = self.get_user_recommendations(user, 10)
            recommended_items = set([rec['product_id'] for rec in recommendations])
            
            # 计算精确率和召回率
            if len(recommended_items) > 0:
                precision = len(test_items & recommended_items) / len(recommended_items)
                precision_scores.append(precision)
            
            if len(test_items) > 0:
                recall = len(test_items & recommended_items) / len(test_items)
                recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        
        print(f"=== 推荐系统评估结果 ===")
        print(f"平均精确率: {avg_precision:.3f}")
        print(f"平均召回率: {avg_recall:.3f}")
        print(f"F1分数: {2 * avg_precision * avg_recall / (avg_precision + avg_recall):.3f}")
        
        return avg_precision, avg_recall
    
    def visualize_recommendations(self, customer_id):
        """可视化推荐结果"""
        # 获取两种推荐结果
        cf_recommendations = self.get_user_recommendations(customer_id, 5)
        cb_recommendations = self.content_based_recommendations(customer_id, 5)
        
        if not cf_recommendations and not cb_recommendations:
            print(f"无法为客户 {customer_id} 生成推荐")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 协同过滤推荐
        if cf_recommendations:
            cf_categories = [rec['category'] for rec in cf_recommendations]
            cf_scores = [rec['recommendation_score'] for rec in cf_recommendations]
            
            axes[0].barh(range(len(cf_categories)), cf_scores)
            axes[0].set_yticks(range(len(cf_categories)))
            axes[0].set_yticklabels(cf_categories)
            axes[0].set_xlabel('推荐分数')
            axes[0].set_title('协同过滤推荐')
        
        # 基于内容推荐
        if cb_recommendations:
            cb_categories = [rec['category'] for rec in cb_recommendations]
            cb_scores = [rec['similarity_score'] for rec in cb_recommendations]
            
            axes[1].barh(range(len(cb_categories)), cb_scores)
            axes[1].set_yticks(range(len(cb_categories)))
            axes[1].set_yticklabels(cb_categories)
            axes[1].set_xlabel('相似度分数')
            axes[1].set_title('基于内容推荐')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细推荐
        print(f"\n=== 为客户 {customer_id} 的推荐结果 ===")
        
        if cf_recommendations:
            print("\n协同过滤推荐:")
            for i, rec in enumerate(cf_recommendations, 1):
                print(f"{i}. {rec['category']} - 价格: ¥{rec['price']:.2f} - 评分: {rec['rating']} - 推荐分数: {rec['recommendation_score']:.3f}")
        
        if cb_recommendations:
            print("\n基于内容推荐:")
            for i, rec in enumerate(cb_recommendations, 1):
                print(f"{i}. {rec['category']} - 价格: ¥{rec['price']:.2f} - 评分: {rec['rating']} - 相似度: {rec['similarity_score']:.3f}")
```

## 任务4：客户流失预测

### 目标
使用随机森林预测客户流失风险。

### 实现代码

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

class ChurnPrediction:
    """客户流失预测"""
    
    def __init__(self, customers_df, behavior_df, transactions_df):
        self.customers_df = customers_df
        self.behavior_df = behavior_df
        self.transactions_df = transactions_df
        self.feature_data = None
        self.model = None
        
    def define_churn(self, days_threshold=90):
        """定义流失客户"""
        # 基于最后购买时间定义流失
        current_date = datetime.now()
        
        churn_labels = []
        for _, customer in self.behavior_df.iterrows():
            days_since_last = customer['days_since_last_order']
            
            # 如果超过阈值天数未购买，则认为流失
            is_churned = 1 if days_since_last > days_threshold else 0
            churn_labels.append(is_churned)
        
        self.behavior_df['is_churned'] = churn_labels
        
        churn_rate = np.mean(churn_labels)
        print(f"流失率: {churn_rate:.2%} (阈值: {days_threshold}天)")
        
        return churn_labels
    
    def create_churn_features(self):
        """创建流失预测特征"""
        # 合并客户基本信息和行为数据
        merged_data = self.customers_df.merge(self.behavior_df, on='customer_id')
        
        # 计算额外特征
        # 1. 客户生命周期
        current_date = datetime.now()
        merged_data['customer_lifetime'] = (
            current_date - merged_data['register_date']
        ).dt.days
        
        # 2. 购买频率
        merged_data['purchase_frequency'] = (
            merged_data['total_orders'] / merged_data['customer_lifetime']
        ).fillna(0)
        
        # 3. 平均订单间隔
        merged_data['avg_order_interval'] = (
            merged_data['customer_lifetime'] / merged_data['total_orders']
        ).fillna(999)
        
        # 4. 消费趋势（最近消费 vs 历史平均）
        # 这里简化处理，实际应该计算时间序列趋势
        merged_data['spending_trend'] = np.random.normal(0, 1, len(merged_data))
        
        # 选择特征
        feature_columns = [
            'age', 'income', 'city_tier', 'total_orders', 'total_amount',
            'avg_order_value', 'days_since_last_order', 'customer_lifetime',
            'purchase_frequency', 'avg_order_interval', 'spending_trend'
        ]
        
        # 处理性别特征
        merged_data['gender_encoded'] = merged_data['gender'].map({'M': 1, 'F': 0})
        feature_columns.append('gender_encoded')
        
        # 处理最喜欢的类别
        category_dummies = pd.get_dummies(
            merged_data['favorite_category'], 
            prefix='fav_category'
        )
        
        # 合并所有特征
        X = pd.concat([
            merged_data[feature_columns],
            category_dummies
        ], axis=1)
        
        y = merged_data['is_churned']
        
        self.feature_data = {
            'X': X,
            'y': y,
            'feature_names': X.columns.tolist(),
            'customer_ids': merged_data['customer_id']
        }
        
        return X, y
    
    def train_churn_model(self):
        """训练流失预测模型"""
        if self.feature_data is None:
            self.define_churn()
            self.create_churn_features()
        
        X, y = self.feature_data['X'], self.feature_data['y']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
        
        # 评估模型
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        auc_score = roc_auc_score(y_test, y_pred_proba_test)
        
        print("=== 流失预测模型性能 ===")
        print(f"训练集准确率: {train_accuracy:.3f}")
        print(f"测试集准确率: {test_accuracy:.3f}")
        print(f"AUC分数: {auc_score:.3f}")
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_test))
        
        return {
            'model': self.model,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_test': y_pred_test,
            'y_pred_proba_test': y_pred_proba_test
        }
    
    def visualize_churn_analysis(self, results):
        """可视化流失分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵
        cm = confusion_matrix(results['y_test'], results['y_pred_test'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')
        
        # 2. ROC曲线
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba_test'])
        auc_score = roc_auc_score(results['y_test'], results['y_pred_proba_test'])
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='随机分类器')
        axes[0, 1].set_xlabel('假正率')
        axes[0, 1].set_ylabel('真正率')
        axes[0, 1].set_title('ROC曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 特征重要性
        importances = self.model.feature_importances_
        feature_names = self.feature_data['feature_names']
        indices = np.argsort(importances)[::-1][:10]  # 前10个重要特征
        
        axes[1, 0].bar(range(len(indices)), importances[indices])
        axes[1, 0].set_title('特征重要性 (前10)')
        axes[1, 0].set_xlabel('特征')
        axes[1, 0].set_ylabel('重要性')
        axes[1, 0].set_xticks(range(len(indices)))
        axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # 4. 流失概率分布
        axes[1, 1].hist(results['y_pred_proba_test'], bins=20, alpha=0.7, 
                       label='流失概率分布')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', 
                          label='分类阈值')
        axes[1, 1].set_xlabel('流失概率')
        axes[1, 1].set_ylabel('客户数量')
        axes[1, 1].set_title('客户流失概率分布')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def identify_high_risk_customers(self, top_n=50):
        """识别高风险流失客户"""
        if self.model is None:
            print("请先训练模型")
            return
        
        X = self.feature_data['X']
        customer_ids = self.feature_data['customer_ids']
        
        # 预测所有客户的流失概率
        churn_probabilities = self.model.predict_proba(X)[:, 1]
        
        # 创建结果DataFrame
        risk_analysis = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_probabilities
        })
        
        # 合并客户信息
        risk_analysis = risk_analysis.merge(
            self.customers_df[['customer_id', 'age', 'income']], 
            on='customer_id'
        )
        risk_analysis = risk_analysis.merge(
            self.behavior_df[['customer_id', 'total_amount', 'days_since_last_order']], 
            on='customer_id'
        )
        
        # 排序并获取高风险客户
        high_risk_customers = risk_analysis.nlargest(top_n, 'churn_probability')
        
        print(f"=== 前{top_n}名高风险流失客户 ===")
        print(high_risk_customers[[
            'customer_id', 'churn_probability', 'total_amount', 
            'days_since_last_order', 'age', 'income'
        ]].round(3))
        
        return high_risk_customers

## 任务5：关联规则分析

### 目标
使用Apriori算法发现商品之间的关联规律，为交叉销售提供支持。

### 实现代码

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class MarketBasketAnalysis:
    """购物篮分析"""
    
    def __init__(self, transactions_df, products_df):
        self.transactions_df = transactions_df
        self.products_df = products_df
        self.basket_data = None
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_basket_data(self):
        """准备购物篮数据"""
        # 合并交易和商品数据
        merged_data = self.transactions_df.merge(self.products_df, on='product_id')
        
        # 按交易分组，获取每个交易的商品类别
        basket_data = merged_data.groupby('transaction_id')['category'].apply(list).reset_index()
        
        # 转换为适合关联规则分析的格式
        transactions = basket_data['category'].tolist()
        
        # 使用TransactionEncoder进行编码
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.basket_data = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"购物篮数据准备完成: {len(self.basket_data)} 个交易, {len(te.columns_)} 个商品类别")
        
        return self.basket_data
    
    def find_frequent_itemsets(self, min_support=0.01):
        """发现频繁项集"""
        if self.basket_data is None:
            self.prepare_basket_data()
        
        # 使用Apriori算法找到频繁项集
        self.frequent_itemsets = apriori(
            self.basket_data, 
            min_support=min_support, 
            use_colnames=True
        )
        
        print(f"发现 {len(self.frequent_itemsets)} 个频繁项集 (最小支持度: {min_support})")
        
        # 显示前10个频繁项集
        print("\n=== 前10个频繁项集 ===")
        top_itemsets = self.frequent_itemsets.nlargest(10, 'support')
        for _, row in top_itemsets.iterrows():
            items = ', '.join(list(row['itemsets']))
            print(f"{items}: 支持度 = {row['support']:.3f}")
        
        return self.frequent_itemsets
    
    def generate_association_rules(self, min_confidence=0.3):
        """生成关联规则"""
        if self.frequent_itemsets is None:
            self.find_frequent_itemsets()
        
        # 生成关联规则
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        # 添加提升度筛选
        self.rules = self.rules[self.rules['lift'] > 1]
        
        print(f"生成 {len(self.rules)} 条关联规则 (最小置信度: {min_confidence})")
        
        return self.rules
    
    def analyze_top_rules(self, top_n=10):
        """分析顶级关联规则"""
        if self.rules is None:
            self.generate_association_rules()
        
        # 按置信度排序
        top_rules = self.rules.nlargest(top_n, 'confidence')
        
        print(f"\n=== 前{top_n}条关联规则 (按置信度排序) ===")
        for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            
            print(f"{i}. {antecedents} → {consequents}")
            print(f"   支持度: {rule['support']:.3f}")
            print(f"   置信度: {rule['confidence']:.3f}")
            print(f"   提升度: {rule['lift']:.3f}")
            print()
        
        return top_rules
    
    def visualize_rules(self):
        """可视化关联规则"""
        if self.rules is None:
            self.generate_association_rules()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 支持度 vs 置信度散点图
        scatter = axes[0, 0].scatter(
            self.rules['support'], 
            self.rules['confidence'],
            c=self.rules['lift'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[0, 0].set_xlabel('支持度')
        axes[0, 0].set_ylabel('置信度')
        axes[0, 0].set_title('关联规则分布 (颜色表示提升度)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. 提升度分布
        axes[0, 1].hist(self.rules['lift'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=1, color='red', linestyle='--', label='提升度=1')
        axes[0, 1].set_xlabel('提升度')
        axes[0, 1].set_ylabel('规则数量')
        axes[0, 1].set_title('提升度分布')
        axes[0, 1].legend()
        
        # 3. 置信度分布
        axes[1, 0].hist(self.rules['confidence'], bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('置信度')
        axes[1, 0].set_ylabel('规则数量')
        axes[1, 0].set_title('置信度分布')
        
        # 4. 支持度分布
        axes[1, 1].hist(self.rules['support'], bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('支持度')
        axes[1, 1].set_ylabel('规则数量')
        axes[1, 1].set_title('支持度分布')
        
        plt.tight_layout()
        plt.show()
    
    def get_recommendations_for_category(self, category, top_n=5):
        """为指定商品类别推荐关联商品"""
        if self.rules is None:
            self.generate_association_rules()
        
        # 找到以该类别为前件的规则
        category_rules = self.rules[
            self.rules['antecedents'].apply(lambda x: category in x)
        ]
        
        if len(category_rules) == 0:
            print(f"没有找到以 '{category}' 为前件的关联规则")
            return []
        
        # 按置信度排序
        top_recommendations = category_rules.nlargest(top_n, 'confidence')
        
        print(f"\n=== 购买 '{category}' 的客户还可能购买 ===")
        recommendations = []
        
        for _, rule in top_recommendations.iterrows():
            consequents = list(rule['consequents'])
            for item in consequents:
                if item != category:
                    recommendations.append({
                        'item': item,
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
                    print(f"- {item} (置信度: {rule['confidence']:.3f}, 提升度: {rule['lift']:.3f})")
        
        return recommendations
```

## 项目整合与部署

### 完整的电商分析系统

```python
class EcommerceAnalyticsSystem:
    """电商数据分析系统整合"""
    
    def __init__(self):
        self.data_loaded = False
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.behavior_df = None
        
        # 分析模块
        self.segmentation = None
        self.forecast = None
        self.recommendation = None
        self.churn_prediction = None
        self.market_basket = None
        
    def load_data(self):
        """加载数据"""
        print("正在生成电商数据集...")
        (
            self.customers_df, 
            self.products_df, 
            self.transactions_df, 
            self.behavior_df
        ) = create_ecommerce_dataset()
        
        self.data_loaded = True
        print("数据加载完成！")
        
    def initialize_modules(self):
        """初始化分析模块"""
        if not self.data_loaded:
            self.load_data()
        
        print("正在初始化分析模块...")
        
        # 初始化各个分析模块
        self.segmentation = CustomerSegmentation(
            self.customers_df, self.behavior_df
        )
        
        self.forecast = SalesForecast(
            self.transactions_df, self.products_df
        )
        
        self.recommendation = RecommendationSystem(
            self.transactions_df, self.products_df, self.customers_df
        )
        
        self.churn_prediction = ChurnPrediction(
            self.customers_df, self.behavior_df, self.transactions_df
        )
        
        self.market_basket = MarketBasketAnalysis(
            self.transactions_df, self.products_df
        )
        
        print("分析模块初始化完成！")
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        if not self.data_loaded:
            self.initialize_modules()
        
        print("\n" + "="*50)
        print("开始电商数据分析")
        print("="*50)
        
        # 1. 数据探索
        print("\n1. 数据探索分析")
        eda = EcommerceEDA(
            self.customers_df, self.products_df, 
            self.transactions_df, self.behavior_df
        )
        eda.basic_statistics()
        
        # 2. 客户细分
        print("\n2. 客户细分分析")
        optimal_k = self.segmentation.find_optimal_clusters()
        cluster_labels, cluster_summary = self.segmentation.perform_clustering(optimal_k)
        self.segmentation.visualize_clusters()
        
        # 3. 销量预测
        print("\n3. 销量预测分析")
        forecast_results, X_test, y_test = self.forecast.train_models()
        self.forecast.visualize_predictions(forecast_results, X_test, y_test)
        
        # 4. 推荐系统
        print("\n4. 推荐系统分析")
        self.recommendation.train_collaborative_filtering()
        
        # 选择一个示例客户进行推荐
        sample_customer = self.customers_df.sample(1)['customer_id'].iloc[0]
        self.recommendation.visualize_recommendations(sample_customer)
        
        # 5. 流失预测
        print("\n5. 客户流失预测")
        self.churn_prediction.define_churn()
        churn_results = self.churn_prediction.train_churn_model()
        self.churn_prediction.visualize_churn_analysis(churn_results)
        
        # 6. 关联规则分析
        print("\n6. 关联规则分析")
        self.market_basket.find_frequent_itemsets()
        self.market_basket.generate_association_rules()
        self.market_basket.analyze_top_rules()
        
        print("\n" + "="*50)
        print("电商数据分析完成！")
        print("="*50)
    
    def generate_business_report(self):
        """生成业务报告"""
        print("\n" + "="*60)
        print("电商数据分析业务报告")
        print("="*60)
        
        # 客户洞察
        print("\n【客户洞察】")
        if self.segmentation and hasattr(self.segmentation, 'merged_df'):
            total_customers = len(self.customers_df)
            active_customers = self.behavior_df['is_active'].sum()
            avg_order_value = self.behavior_df['avg_order_value'].mean()
            
            print(f"- 总客户数: {total_customers:,}")
            print(f"- 活跃客户数: {active_customers:,} ({active_customers/total_customers*100:.1f}%)")
            print(f"- 平均订单价值: ¥{avg_order_value:.2f}")
        
        # 销售洞察
        print("\n【销售洞察】")
        total_revenue = self.transactions_df['total_amount'].sum()
        total_orders = len(self.transactions_df)
        avg_order_size = self.transactions_df['total_amount'].mean()
        
        print(f"- 总收入: ¥{total_revenue:,.2f}")
        print(f"- 总订单数: {total_orders:,}")
        print(f"- 平均订单金额: ¥{avg_order_size:.2f}")
        
        # 商品洞察
        print("\n【商品洞察】")
        top_categories = self.transactions_df.merge(
            self.products_df, on='product_id'
        )['category'].value_counts().head(3)
        
        print("- 热销类别:")
        for category, count in top_categories.items():
            print(f"  {category}: {count} 次购买")
        
        # 业务建议
        print("\n【业务建议】")
        print("1. 客户细分策略:")
        print("   - 针对高价值客户提供VIP服务")
        print("   - 对潜力客户进行精准营销")
        print("   - 挽回流失风险客户")
        
        print("\n2. 商品推荐策略:")
        print("   - 基于协同过滤的个性化推荐")
        print("   - 利用关联规则进行交叉销售")
        print("   - 优化商品组合和定价")
        
        print("\n3. 运营优化建议:")
        print("   - 提升客户活跃度和复购率")
        print("   - 优化库存管理和供应链")
        print("   - 加强数据驱动的决策制定")

# 运行完整系统
def run_ecommerce_analysis():
    """运行电商分析系统"""
    # 创建系统实例
    system = EcommerceAnalyticsSystem()
    
    # 运行完整分析
    system.run_full_analysis()
    
    # 生成业务报告
    system.generate_business_report()
    
    return system

# 使用示例
if __name__ == "__main__":
    # 运行电商分析系统
    analytics_system = run_ecommerce_analysis()
    
    print("\n分析完成！您可以进一步探索各个模块的功能。")
```

## 学习总结

### 项目收获

通过这个综合实践项目，你将掌握：

1. **端到端的机器学习项目流程**
   - 数据生成和预处理
   - 特征工程和选择
   - 模型训练和评估
   - 结果可视化和解释

2. **多种算法的实际应用**
   - K-means聚类：客户细分
   - 线性回归和随机森林：销量预测
   - K近邻：推荐系统
   - 随机森林分类：流失预测
   - Apriori算法：关联规则挖掘

3. **业务理解和数据科学思维**
   - 将业务问题转化为技术问题
   - 选择合适的算法和评估指标
   - 解释模型结果并提供业务建议

### 扩展方向

1. **深度学习应用**
   - 使用神经网络进行更复杂的预测
   - 实现深度推荐系统
   - 应用自然语言处理分析用户评论

2. **实时系统构建**
   - 构建实时推荐API
   - 实现流式数据处理
   - 部署模型到生产环境

3. **高级分析技术**
   - 时间序列分析
   - 因果推断
   - A/B测试设计和分析

### 思考题

1. 如何处理数据不平衡问题（如流失客户比例很小）？
2. 如何评估推荐系统的长期效果？
3. 如何将多个模型的结果整合为统一的业务决策？
4. 如何处理新客户的冷启动问题？
5. 如何确保模型的公平性和可解释性？

## 项目总结

本综合实践项目通过构建电商数据分析系统，展示了传统机器学习算法在实际业务场景中的应用。项目涵盖了数据科学的完整流程，从数据生成、探索分析到模型构建和业务应用，为你提供了宝贵的实践经验。

通过这个项目，你不仅学会了如何使用各种算法，更重要的是培养了数据科学思维和解决实际问题的能力。这些技能将为你在AI和数据科学领域的进一步发展奠定坚实基础。