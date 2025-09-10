import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")

def plot_class_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        y=df['category'],
        order=df['category'].value_counts().index,
        palette="viridis",
        ax=ax
    )
    ax.set_title('Object Class Distribution', fontsize=16)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('Object Category', fontsize=12)
    plt.tight_layout()
    return fig

def plot_box_area_by_class(df: pd.DataFrame, selected_class: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    class_df = df[df['category'] == selected_class]
    sns.histplot(class_df['area'], bins=50, log_scale=True, ax=ax, color='skyblue')
    ax.set_title(f"Bounding Box Area Distribution for '{selected_class}' (Log Scale)", fontsize=14)
    ax.set_xlabel('Area (pixels^2)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    return fig

def plot_distribution_by_condition(df: pd.DataFrame, condition: str = 'weather'):
    fig, ax = plt.subplots(figsize=(12, 8))
    conditional_dist = df.groupby([condition, 'category']).size().unstack(fill_value=0)
    conditional_norm = conditional_dist.div(conditional_dist.sum(axis=1), axis=0)
    sns.heatmap(
        conditional_norm.T,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title(f'Normalized Class Distribution by {condition.capitalize()}', fontsize=16)
    ax.set_xlabel(condition.capitalize(), fontsize=12)
    ax.set_ylabel('Object Category', fontsize=12)
    plt.tight_layout()
    return fig

def get_annotated_image(image_name: str, images_dir: Path, df: pd.DataFrame):
    image_path = images_dir / f"{image_name}.jpg"
    if not image_path.exists():
        return None
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_labels = df[df['image_name'] == image_name]
    
    for _, row in image_labels.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        color = (36, 255, 12)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, row['category'], (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    return image
