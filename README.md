# Pythonでグラフを作成する完全ガイド

このチュートリアルでは、Pythonを使用してさまざまな種類のグラフを作成する方法を、実際のサンプルコードとともに説明します。主にMatplotlibとSeabornライブラリを使用します。

## 目次
1. [必要なライブラリのインストール](#必要なライブラリのインストール)
2. [基本的なグラフの種類](#基本的なグラフの種類)
   - [折れ線グラフ（Line Plot）](#1-折れ線グラフline-plot)
   - [棒グラフ（Bar Chart）](#2-棒グラフbar-chart)
   - [散布図（Scatter Plot）](#3-散布図scatter-plot)
   - [ヒストグラム（Histogram）](#4-ヒストグラムhistogram)
   - [円グラフ（Pie Chart）](#5-円グラフpie-chart)
   - [ヒートマップ（Heatmap）](#6-ヒートマップheatmap)
   - [箱ひげ図（Box Plot）](#7-箱ひげ図box-plot)
   - [3Dプロット](#8-3dプロット)

## 必要なライブラリのインストール

まず、必要なライブラリをインストールします：

```bash
pip install matplotlib seaborn pandas numpy
```

## 基本的なインポート

すべてのサンプルで以下のインポートを使用します：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 日本語フォントの設定（必要に応じて）
plt.rcParams['font.family'] = 'DejaVu Sans'
# Windows環境では以下を使用
# plt.rcParams['font.family'] = 'MS Gothic'

# グラフのスタイル設定
sns.set_style("whitegrid")
```

## 基本的なグラフの種類

### 1. 折れ線グラフ（Line Plot）

時系列データや連続的な変化を表現するのに適しています。

#### 基本的な折れ線グラフ

```python
# データの準備
x = np.linspace(0, 10, 100)
y = np.sin(x)

# グラフの作成
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('X軸')
plt.ylabel('Y軸')
plt.title('基本的な折れ線グラフ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 複数の折れ線グラフ

```python
# データの準備
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * 0.5

# グラフの作成
plt.figure(figsize=(12, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.plot(x, y3, label='0.5*sin(x)', color='green', linewidth=2, linestyle=':')
plt.xlabel('X軸')
plt.ylabel('Y軸')
plt.title('複数の折れ線グラフ')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 時系列データの折れ線グラフ

```python
# データの準備
dates = pd.date_range('2024-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100

# DataFrameの作成
df = pd.DataFrame({'日付': dates, '値': values})

# グラフの作成
plt.figure(figsize=(14, 6))
plt.plot(df['日付'], df['値'], color='darkblue', linewidth=1.5)
plt.xlabel('日付')
plt.ylabel('値')
plt.title('時系列データの推移')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 2. 棒グラフ（Bar Chart）

カテゴリ別のデータを比較するのに適しています。

#### 基本的な棒グラフ

```python
# データの準備
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

# グラフの作成
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color='skyblue', edgecolor='navy', linewidth=2)

# 値をバーの上に表示
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom')

plt.xlabel('カテゴリ')
plt.ylabel('値')
plt.title('基本的な棒グラフ')
plt.ylim(0, max(values) * 1.1)
plt.show()
```

#### 横向き棒グラフ

```python
# データの準備
categories = ['製品A', '製品B', '製品C', '製品D', '製品E']
sales = [156, 234, 189, 267, 198]

# グラフの作成
plt.figure(figsize=(10, 6))
bars = plt.barh(categories, sales, color='lightgreen', edgecolor='darkgreen')

# 値をバーの右に表示
for bar in bars:
    width = bar.get_width()
    plt.text(width + 3, bar.get_y() + bar.get_height()/2.,
             f'{width}', ha='left', va='center')

plt.xlabel('売上（万円）')
plt.ylabel('製品')
plt.title('製品別売上（横棒グラフ）')
plt.xlim(0, max(sales) * 1.1)
plt.show()
```

#### グループ化された棒グラフ

```python
# データの準備
categories = ['1月', '2月', '3月', '4月', '5月']
product_A = [20, 35, 30, 35, 27]
product_B = [25, 32, 34, 20, 25]
product_C = [15, 20, 35, 30, 30]

x = np.arange(len(categories))
width = 0.25

# グラフの作成
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, product_A, width, label='製品A', color='#FF6B6B')
rects2 = ax.bar(x, product_B, width, label='製品B', color='#4ECDC4')
rects3 = ax.bar(x + width, product_C, width, label='製品C', color='#45B7D1')

ax.set_xlabel('月')
ax.set_ylabel('売上（万円）')
ax.set_title('月別・製品別売上')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 3. 散布図（Scatter Plot）

2つの変数の関係性を表現するのに適しています。

#### 基本的な散布図

```python
# データの準備
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# グラフの作成
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=1)
plt.xlabel('X値')
plt.ylabel('Y値')
plt.title('基本的な散布図')
plt.grid(True, alpha=0.3)
plt.show()
```

#### カラーマップを使用した散布図

```python
# データの準備
np.random.seed(42)
n = 150
x = np.random.randn(n)
y = np.random.randn(n)
colors = np.random.randn(n)
sizes = np.abs(np.random.randn(n)) * 100

# グラフの作成
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label='カラー値')
plt.xlabel('X値')
plt.ylabel('Y値')
plt.title('カラーマップとサイズを使用した散布図')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 回帰線付き散布図

```python
# データの準備
np.random.seed(42)
x = np.random.uniform(0, 10, 50)
y = 2.5 * x + np.random.normal(0, 2, 50)

# 回帰線の計算
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# グラフの作成
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.7, s=100, c='coral', edgecolors='darkred', linewidth=1, label='データ点')
plt.plot(x, p(x), "r-", linewidth=2, label=f'回帰線: y={z[0]:.2f}x+{z[1]:.2f}')
plt.xlabel('X値')
plt.ylabel('Y値')
plt.title('回帰線付き散布図')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. ヒストグラム（Histogram）

データの分布を表現するのに適しています。

#### 基本的なヒストグラム

```python
# データの準備
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# グラフの作成
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# 平均値と中央値の線を追加
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'平均値: {data.mean():.2f}')
plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label=f'中央値: {np.median(data):.2f}')

plt.xlabel('値')
plt.ylabel('頻度')
plt.title('正規分布のヒストグラム')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

#### 複数のヒストグラム（重ね合わせ）

```python
# データの準備
np.random.seed(42)
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(130, 20, 1000)

# グラフの作成
plt.figure(figsize=(12, 6))
plt.hist(data1, bins=30, alpha=0.5, label='グループA', color='blue', edgecolor='black')
plt.hist(data2, bins=30, alpha=0.5, label='グループB', color='red', edgecolor='black')

plt.xlabel('値')
plt.ylabel('頻度')
plt.title('複数グループのヒストグラム比較')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

#### 2次元ヒストグラム

```python
# データの準備
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

# グラフの作成
plt.figure(figsize=(10, 8))
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(label='頻度')
plt.xlabel('X値')
plt.ylabel('Y値')
plt.title('2次元ヒストグラム')
plt.show()
```

### 5. 円グラフ（Pie Chart）

全体に対する各部分の割合を表現するのに適しています。

#### 基本的な円グラフ

```python
# データの準備
labels = ['Python', 'JavaScript', 'Java', 'C++', 'Others']
sizes = [35, 25, 20, 15, 5]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# グラフの作成
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('プログラミング言語の使用割合')
plt.axis('equal')
plt.show()
```

#### 一部を強調した円グラフ

```python
# データの準備
labels = ['製品A', '製品B', '製品C', '製品D', '製品E']
sizes = [30, 25, 20, 15, 10]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
explode = (0.1, 0, 0, 0, 0)  # 最初の要素を強調

# グラフの作成
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('製品別売上構成比（製品Aを強調）')
plt.axis('equal')
plt.show()
```

#### ドーナツグラフ

```python
# データの準備
labels = ['カテゴリA', 'カテゴリB', 'カテゴリC', 'カテゴリD']
sizes = [40, 30, 20, 10]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# グラフの作成
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=90,
                                    pctdistance=0.85)

# 中心に白い円を作成してドーナツ型にする
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.title('ドーナツグラフのサンプル')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

### 6. ヒートマップ（Heatmap）

2次元データの値を色で表現するのに適しています。

#### 基本的なヒートマップ

```python
# データの準備
np.random.seed(42)
data = np.random.randn(10, 12)

# グラフの作成
plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('基本的なヒートマップ')
plt.xlabel('列')
plt.ylabel('行')
plt.show()
```

#### 相関行列のヒートマップ

```python
# データの準備
np.random.seed(42)
df = pd.DataFrame({
    '変数A': np.random.randn(100),
    '変数B': np.random.randn(100),
    '変数C': np.random.randn(100),
    '変数D': np.random.randn(100),
    '変数E': np.random.randn(100)
})

# 相関行列の計算
correlation_matrix = df.corr()

# グラフの作成
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
plt.title('相関行列のヒートマップ')
plt.tight_layout()
plt.show()
```

#### カスタムカラーマップのヒートマップ

```python
# データの準備
months = ['1月', '2月', '3月', '4月', '5月', '6月',
          '7月', '8月', '9月', '10月', '11月', '12月']
products = ['製品A', '製品B', '製品C', '製品D', '製品E']

# ランダムな売上データを生成
np.random.seed(42)
sales_data = np.random.randint(50, 200, size=(len(products), len(months)))

# DataFrameの作成
df_sales = pd.DataFrame(sales_data, index=products, columns=months)

# グラフの作成
plt.figure(figsize=(14, 6))
sns.heatmap(df_sales, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={"label": "売上（万円）"})
plt.title('月別・製品別売上ヒートマップ')
plt.ylabel('製品')
plt.xlabel('月')
plt.tight_layout()
plt.show()
```

### 7. 箱ひげ図（Box Plot）

データの分布と外れ値を表現するのに適しています。

#### 基本的な箱ひげ図

```python
# データの準備
np.random.seed(42)
data = [np.random.normal(100, 10, 200),
        np.random.normal(90, 20, 200),
        np.random.normal(110, 15, 200),
        np.random.normal(95, 25, 200)]

# グラフの作成
fig, ax = plt.subplots(figsize=(10, 6))
box_plot = ax.boxplot(data, labels=['グループA', 'グループB', 'グループC', 'グループD'],
                       patch_artist=True, notch=True)

# 色の設定
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('値')
plt.title('グループ別データ分布（箱ひげ図）')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

#### Seabornを使用した箱ひげ図

```python
# データの準備
np.random.seed(42)
df = pd.DataFrame({
    'カテゴリ': np.repeat(['A', 'B', 'C', 'D'], 100),
    '値': np.concatenate([
        np.random.normal(100, 10, 100),
        np.random.normal(90, 20, 100),
        np.random.normal(110, 15, 100),
        np.random.normal(95, 25, 100)
    ])
})

# グラフの作成
plt.figure(figsize=(10, 6))
sns.boxplot(x='カテゴリ', y='値', data=df, palette='Set2')
plt.title('カテゴリ別データ分布（Seaborn箱ひげ図）')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

#### バイオリンプロット

```python
# データの準備（箱ひげ図と同じデータを使用）
np.random.seed(42)
df = pd.DataFrame({
    'カテゴリ': np.repeat(['A', 'B', 'C', 'D'], 100),
    '値': np.concatenate([
        np.random.normal(100, 10, 100),
        np.random.normal(90, 20, 100),
        np.random.normal(110, 15, 100),
        np.random.normal(95, 25, 100)
    ])
})

# グラフの作成
plt.figure(figsize=(10, 6))
sns.violinplot(x='カテゴリ', y='値', data=df, palette='muted', inner='box')
plt.title('カテゴリ別データ分布（バイオリンプロット）')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

### 8. 3Dプロット

3次元データを表現するのに適しています。

#### 3D散布図

```python
from mpl_toolkits.mplot3d import Axes3D

# データの準備
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
colors = np.random.randn(n)

# グラフの作成
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.6)

ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')
ax.set_title('3D散布図')

plt.colorbar(scatter, ax=ax, pad=0.1)
plt.show()
```

#### 3D曲面プロット

```python
from mpl_toolkits.mplot3d import Axes3D

# データの準備
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# グラフの作成
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0,
                       antialiased=True, alpha=0.8)

ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')
ax.set_title('3D曲面プロット')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
```

#### 3Dワイヤーフレーム

```python
from mpl_toolkits.mplot3d import Axes3D

# データの準備
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# グラフの作成
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5)

ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')
ax.set_title('3Dワイヤーフレーム')
plt.show()
```

## 高度なテクニック

### サブプロット（複数のグラフを1つの図に配置）

```python
# データの準備
np.random.seed(42)
x = np.linspace(0, 10, 100)

# グラフの作成
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：折れ線グラフ
axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('折れ線グラフ')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# 右上：散布図
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), alpha=0.5)
axes[0, 1].set_title('散布図')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].grid(True, alpha=0.3)

# 左下：ヒストグラム
axes[1, 0].hist(np.random.normal(100, 15, 1000), bins=30, color='green', alpha=0.7)
axes[1, 0].set_title('ヒストグラム')
axes[1, 0].set_xlabel('値')
axes[1, 0].set_ylabel('頻度')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 右下：棒グラフ
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values, color='orange')
axes[1, 1].set_title('棒グラフ')
axes[1, 1].set_xlabel('カテゴリ')
axes[1, 1].set_ylabel('値')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('サブプロットの例', fontsize=16)
plt.tight_layout()
plt.show()
```

### アニメーション

```python
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# データの準備
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('正弦波のアニメーション')
ax.grid(True, alpha=0.3)

line, = ax.plot([], [], 'b-', linewidth=2)

def init():
    line.set_data([], [])
    return line,

def animate(frame):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x + frame/10)
    line.set_data(x, y)
    return line,

# アニメーションの作成
anim = FuncAnimation(fig, animate, init_func=init, frames=100,
                    interval=50, blit=True)

plt.show()

# Jupyter Notebookで表示する場合
# HTML(anim.to_jshtml())
```

### カスタムスタイル

```python
# Seabornのスタイルを使用
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, style in enumerate(styles):
    with sns.axes_style(style):
        axes[idx].plot(np.random.randn(100).cumsum())
        axes[idx].set_title(f'Style: {style}')
        axes[idx].set_xlabel('Index')
        axes[idx].set_ylabel('Value')

# 最後の軸を非表示
axes[-1].set_visible(False)

plt.suptitle('Seabornスタイルの比較', fontsize=16)
plt.tight_layout()
plt.show()
```

## グラフの保存

作成したグラフをファイルとして保存する方法：

```python
# 高解像度での保存
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.random.randn(100).cumsum())
ax.set_title('保存するグラフ')
ax.set_xlabel('時間')
ax.set_ylabel('値')
ax.grid(True, alpha=0.3)

# PNG形式で保存（高解像度）
plt.savefig('graph.png', dpi=300, bbox_inches='tight')

# PDF形式で保存（ベクター形式）
plt.savefig('graph.pdf', bbox_inches='tight')

# SVG形式で保存（ベクター形式）
plt.savefig('graph.svg', bbox_inches='tight')

plt.show()
```

## 実践的な例：ダッシュボード風レイアウト

```python
# データの準備
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')

# 図の作成
fig = plt.figure(figsize=(16, 10))
fig.suptitle('2024年度 売上ダッシュボード', fontsize=20, fontweight='bold')

# GridSpecを使用したレイアウト
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 時系列グラフ（上部全体）
ax1 = fig.add_subplot(gs[0, :])
sales = np.cumsum(np.random.randn(365)) + 1000
ax1.plot(dates, sales, color='#2E86AB', linewidth=2)
ax1.fill_between(dates, sales, alpha=0.3, color='#2E86AB')
ax1.set_title('年間売上推移', fontsize=14, fontweight='bold')
ax1.set_xlabel('日付')
ax1.set_ylabel('売上（万円）')
ax1.grid(True, alpha=0.3)

# 2. 月別売上（棒グラフ）
ax2 = fig.add_subplot(gs[1, 0])
months = ['1月', '2月', '3月', '4月', '5月', '6月']
monthly_sales = np.random.randint(80, 150, 6)
bars = ax2.bar(months, monthly_sales, color='#A23B72')
ax2.set_title('上半期月別売上', fontsize=12, fontweight='bold')
ax2.set_ylabel('売上（万円）')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom', fontsize=10)

# 3. カテゴリ別売上（円グラフ）
ax3 = fig.add_subplot(gs[1, 1])
categories = ['製品A', '製品B', '製品C', '製品D']
sizes = [35, 25, 25, 15]
colors_pie = ['#F18F01', '#C73E1D', '#6A994E', '#A7C957']
ax3.pie(sizes, labels=categories, colors=colors_pie, autopct='%1.1f%%',
        startangle=90)
ax3.set_title('カテゴリ別売上構成', fontsize=12, fontweight='bold')

# 4. 地域別売上（横棒グラフ）
ax4 = fig.add_subplot(gs[1, 2])
regions = ['関東', '関西', '中部', '九州', '東北']
regional_sales = [450, 380, 290, 210, 170]
bars_h = ax4.barh(regions, regional_sales, color='#55A630')
ax4.set_title('地域別売上', fontsize=12, fontweight='bold')
ax4.set_xlabel('売上（万円）')
for bar in bars_h:
    width = bar.get_width()
    ax4.text(width + 5, bar.get_y() + bar.get_height()/2.,
             f'{width}', ha='left', va='center', fontsize=10)

# 5. 売上分布（ヒストグラム）
ax5 = fig.add_subplot(gs[2, 0])
daily_sales = np.random.normal(100, 20, 365)
ax5.hist(daily_sales, bins=30, color='#7209B7', alpha=0.7, edgecolor='black')
ax5.axvline(daily_sales.mean(), color='red', linestyle='dashed',
            linewidth=2, label=f'平均: {daily_sales.mean():.1f}')
ax5.set_title('日別売上分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('売上（万円）')
ax5.set_ylabel('頻度')
ax5.legend()

# 6. 相関ヒートマップ
ax6 = fig.add_subplot(gs[2, 1:])
metrics = pd.DataFrame({
    '売上': np.random.randn(100),
    '訪問者数': np.random.randn(100),
    '広告費': np.random.randn(100),
    '在庫': np.random.randn(100),
    '従業員数': np.random.randn(100)
})
correlation = metrics.corr()
im = ax6.imshow(correlation, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(np.arange(len(correlation.columns)))
ax6.set_yticks(np.arange(len(correlation.columns)))
ax6.set_xticklabels(correlation.columns, rotation=45, ha='right')
ax6.set_yticklabels(correlation.columns)
ax6.set_title('指標間の相関', fontsize=12, fontweight='bold')

# 値を表示
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        text = ax6.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
```

## まとめ

このチュートリアルでは、Pythonで作成できる主要なグラフの種類とその実装方法を紹介しました。各グラフの特徴を理解し、データの性質と目的に応じて適切なグラフを選択することが重要です。

### グラフ選択のガイドライン

| データの種類 | 推奨グラフ | 用途 |
|------------|-----------|------|
| 時系列データ | 折れ線グラフ | トレンドや変化の把握 |
| カテゴリ比較 | 棒グラフ | 異なるカテゴリ間の値の比較 |
| 2変数の関係 | 散布図 | 相関関係の確認 |
| データ分布 | ヒストグラム、箱ひげ図 | データの分散や外れ値の確認 |
| 構成比 | 円グラフ | 全体に対する各部分の割合 |
| 多次元データ | ヒートマップ | パターンや相関の視覚化 |
| 3次元データ | 3Dプロット | 空間的な関係の表現 |

### ベストプラクティス

1. **適切なグラフタイプの選択**：データの性質と伝えたいメッセージに応じて選択
2. **明確なラベル付け**：軸ラベル、タイトル、凡例を必ず含める
3. **色の使い方**：色覚異常に配慮した配色を使用
4. **データインクレシオ**：不要な装飾を避け、データそのものに焦点を当てる
5. **一貫性**：同じレポート内では統一されたスタイルを使用

### 追加リソース

- [Matplotlib公式ドキュメント](https://matplotlib.org/stable/contents.html)
- [Seaborn公式ドキュメント](https://seaborn.pydata.org/)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [Plotly（インタラクティブなグラフ）](https://plotly.com/python/)

これらのサンプルコードをベースに、自分のデータに合わせてカスタマイズし、効果的なデータビジュアライゼーションを作成してください。