import pandas as pd
from pprint import pprint
from IPython.display import Image
from sklearn.model_selection import train_test_split

def create_table(df, label_column):
    table = {}

    # determine values for the label
    value_counts = df[label_column].value_counts().sort_index()
    table["class_names"] = value_counts.index.to_numpy()
    table["class_counts"] = value_counts.values
    
    # determine probabilities for the features
    for feature in df.drop(label_column, axis=1).columns:
        table[feature] = {}
        # determine counts
        counts = df.groupby(label_column)[feature].value_counts()
        df_counts = counts.unstack(label_column)
        # add one count to avoid "problem of rare values"
        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace=True)
            df_counts += 1

        # calculate probabilities
        df_probabilities = df_counts / df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].to_numpy()
            table[feature][value] = probabilities
            
    return table


def predict_example(row, lookup_table):
    print(row)
    class_estimates = lookup_table["class_counts"]
    for feature in row.index:
        try:
            value = row[feature]
            probabilities = lookup_table[feature][value]
            class_estimates = class_estimates * probabilities

        # skip in case "value" only occurs in test set but not in train set
        # (i.e. "value" is not in "lookup_table")
        except KeyError:
            continue

    index_max_class = class_estimates.argmax()
    prediction = lookup_table["class_names"][index_max_class]
    
    return prediction


#Load data
df = pd.read_csv('../../Data/galaxymorphology/dataset1_sydney.csv')
df_train, df_test = train_test_split(df, test_size = 0.2)
test_labels = df_test.iloc[:, -1]
df_test.drop(columns = ['target'], inplace= True)

lookup_table = create_table(df_train, label_column="target")
pprint(lookup_table)

predictions = df_test.apply(predict_example, axis=1, args=(lookup_table,))
predictions.head()

predictions_correct = predictions == test_labels
accuracy = predictions_correct.mean()
print(f"Accuracy: {accuracy:.3f}")