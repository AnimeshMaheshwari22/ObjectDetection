import pandas as pd
from typing import Dict, Any

def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    return df['category'].value_counts()

def analyze_train_val_split(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_dist = train_df['category'].value_counts(normalize=True).rename("train_dist")
    val_dist = val_df['category'].value_counts(normalize=True).rename("val_dist")
    
    comparison_df = pd.concat([train_dist, val_dist], axis=1)
    comparison_df['difference'] = (comparison_df['train_dist'] - comparison_df['val_dist']).abs()
    
    return comparison_df.sort_values(by='difference', ascending=False)

def find_anomalies_and_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    
    small_boxes = df[df['area'] < 100]
    extreme_aspect_ratios = df[(df['aspect_ratio'] > 5) | (df['aspect_ratio'] < 0.2)]

    occlusion_rate = df.groupby('category')['occluded'].value_counts(normalize=True).unstack().fillna(0)
    truncation_rate = df.groupby('category')['truncated'].value_counts(normalize=True).unstack().fillna(0)

    objects_per_image = df.groupby('image_name').size().sort_values(ascending=False)
    
    analysis_results = {
        "box_size_stats": df.groupby('category')['area'].describe(),
        "small_box_examples": small_boxes.head(),
        "extreme_aspect_ratio_examples": extreme_aspect_ratios.head(),
        "occlusion_rate_per_class": occlusion_rate,
        "truncation_rate_per_class": truncation_rate,
        "most_complex_images": objects_per_image.head(10),
    }
    
    return analysis_results, df
