from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    digits = load_digits()
    x = digits.data
    y = digits.target

    pipeline = Pipeline([
        ('ss', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=300, random_state=42))
    ])
    print(cross_val_score(pipeline, x, y, n_jobs=-1))

    