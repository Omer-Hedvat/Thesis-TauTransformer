import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

def main():
    df = pd.read_csv('data/glass.csv')

    def get_wasserstein_dist(feature, label1, label2):
        dist = wasserstein_distance(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
        return dist

    def flatten(t):
        return [item for sublist in t for item in sublist]

    features = df.columns.drop('label')
    classes = df['label'].unique()
    distances = []
    for feature in features:
        class_dist = []
        for cls_feature1 in classes:
            class_row = [get_wasserstein_dist(feature, cls_feature1, cls_feature2) for cls_feature2 in classes]
            class_dist.append(class_row)
        distances.append(class_dist)

    two_d_mat = [flatten(distances[idx]) for idx in range(len(distances))]


if __name__ == '__main__':
    main()