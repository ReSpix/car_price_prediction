import dill
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def drop_columns(df):
    columns_to_drop = [
        "id",
        "url",
        "region",
        "region_url",
        "price",
        "manufacturer",
        "image_url",
        "description",
        "posting_date",
        "lat",
        "long",
    ]
    return df.drop(columns_to_drop, axis=1)


def filter_outliers(df):
    def calculate_boundaries(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    def cut_outlier(value, boundaries):
        if value < boundaries[0]:
            return round(boundaries[0])
        if value > boundaries[1]:
            return round(boundaries[1])
        return value

    boundaries = calculate_boundaries(df['year'])
    df_copy = df.copy()
    df_copy['year'].apply(lambda x: cut_outlier(x, boundaries))
    return df_copy


def create_new_features(df):
    import pandas
    new_feat_df = pandas.DataFrame()
    new_feat_df['short_model'] = df['model'].apply(
        lambda x: x if pandas.isna(x) else x.lower().split(' ')[0])
    new_feat_df['age_category'] = df['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df.join(new_feat_df)


def main():
    df = pd.read_csv("data/homework.csv")
    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    encode = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(
            dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer,
         make_column_selector(dtype_include=object))
    ])

    preprocessor = Pipeline(steps=[
        ('drop', FunctionTransformer(drop_columns)),
        ('outliers', FunctionTransformer(filter_outliers)),
        ('feat_eng', FunctionTransformer(create_new_features)),
        ('encode', encode)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(
            f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(
        f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    best_pipe.fit(X, y)
    with open('api/cars_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car price category prediction model',
                'author': 'Yakov Yanovich',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


if __name__ == "__main__":
    main()
