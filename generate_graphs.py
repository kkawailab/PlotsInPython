#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
グラフ画像を生成するスクリプト
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 乱数シードを固定
np.random.seed(42)

def create_line_plot():
    """基本的な折れ線グラフ"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Basic Line Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/line_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Line plot saved")

def create_multiple_lines():
    """複数の折れ線グラフ"""
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * 0.5

    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
    plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
    plt.plot(x, y3, label='0.5*sin(x)', color='green', linewidth=2, linestyle=':')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Multiple Line Plots')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig('images/multiple_lines.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Multiple lines plot saved")

def create_bar_chart():
    """基本的な棒グラフ"""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color='skyblue', edgecolor='navy', linewidth=2)

    # 値をバーの上に表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}', ha='center', va='bottom')

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Basic Bar Chart')
    plt.ylim(0, max(values) * 1.1)
    plt.savefig('images/bar_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Bar chart saved")

def create_grouped_bar():
    """グループ化された棒グラフ"""
    categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
    product_A = [20, 35, 30, 35, 27]
    product_B = [25, 32, 34, 20, 25]
    product_C = [15, 20, 35, 30, 30]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, product_A, width, label='Product A', color='#FF6B6B')
    rects2 = ax.bar(x, product_B, width, label='Product B', color='#4ECDC4')
    rects3 = ax.bar(x + width, product_C, width, label='Product C', color='#45B7D1')

    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.set_title('Monthly Sales by Product')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('images/grouped_bar.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Grouped bar chart saved")

def create_scatter_plot():
    """基本的な散布図"""
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=1)
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.title('Basic Scatter Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('images/scatter_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Scatter plot saved")

def create_scatter_with_regression():
    """回帰線付き散布図"""
    x = np.random.uniform(0, 10, 50)
    y = 2.5 * x + np.random.normal(0, 2, 50)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.7, s=100, c='coral', edgecolors='darkred', linewidth=1, label='Data points')
    plt.plot(sorted(x), p(sorted(x)), "r-", linewidth=2, label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/scatter_regression.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Scatter with regression saved")

def create_histogram():
    """基本的なヒストグラム"""
    data = np.random.normal(100, 15, 1000)

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {data.mean():.2f}')
    plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(data):.2f}')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Normal Distribution Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('images/histogram.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Histogram saved")

def create_pie_chart():
    """基本的な円グラフ"""
    labels = ['Python', 'JavaScript', 'Java', 'C++', 'Others']
    sizes = [35, 25, 20, 15, 5]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Programming Language Usage')
    plt.axis('equal')
    plt.savefig('images/pie_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Pie chart saved")

def create_donut_chart():
    """ドーナツグラフ"""
    labels = ['Category A', 'Category B', 'Category C', 'Category D']
    sizes = [40, 30, 20, 10]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.title('Donut Chart Example')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('images/donut_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Donut chart saved")

def create_heatmap():
    """基本的なヒートマップ"""
    data = np.random.randn(10, 12)

    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Basic Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.savefig('images/heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Heatmap saved")

def create_correlation_heatmap():
    """相関行列のヒートマップ"""
    df = pd.DataFrame({
        'Var A': np.random.randn(100),
        'Var B': np.random.randn(100),
        'Var C': np.random.randn(100),
        'Var D': np.random.randn(100),
        'Var E': np.random.randn(100)
    })

    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Correlation heatmap saved")

def create_box_plot():
    """箱ひげ図"""
    data = [np.random.normal(100, 10, 200),
            np.random.normal(90, 20, 200),
            np.random.normal(110, 15, 200),
            np.random.normal(95, 25, 200)]

    fig, ax = plt.subplots(figsize=(10, 6))
    box_plot = ax.boxplot(data, labels=['Group A', 'Group B', 'Group C', 'Group D'],
                           patch_artist=True, notch=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel('Value')
    plt.title('Box Plot - Data Distribution by Group')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('images/box_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Box plot saved")

def create_3d_surface():
    """3D曲面プロット"""
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0,
                           antialiased=True, alpha=0.8)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Surface Plot')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('images/3d_surface.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ 3D surface plot saved")

def create_subplots():
    """サブプロット例"""
    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 左上：折れ線グラフ
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('sin(x)')
    axes[0, 0].grid(True, alpha=0.3)

    # 右上：散布図
    axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), alpha=0.5)
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].grid(True, alpha=0.3)

    # 左下：ヒストグラム
    axes[1, 0].hist(np.random.normal(100, 15, 1000), bins=30, color='green', alpha=0.7)
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 右下：棒グラフ
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    axes[1, 1].bar(categories, values, color='orange')
    axes[1, 1].set_title('Bar Chart')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Subplot Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/subplots.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Subplots saved")

def main():
    """全てのグラフを生成"""
    print("\n=== Generating graphs ===\n")

    create_line_plot()
    create_multiple_lines()
    create_bar_chart()
    create_grouped_bar()
    create_scatter_plot()
    create_scatter_with_regression()
    create_histogram()
    create_pie_chart()
    create_donut_chart()
    create_heatmap()
    create_correlation_heatmap()
    create_box_plot()
    create_3d_surface()
    create_subplots()

    print("\n=== All graphs generated successfully! ===\n")

if __name__ == "__main__":
    main()