import sys
from pathlib import Path

try:
	import pandas as pd
	from sklearn.linear_model import LinearRegression
	import joblib
except ModuleNotFoundError as e:
	print(f"Missing dependency: {e.name}. Install with: pip install -r requirements.txt")
	sys.exit(1)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "datasets" / "workout_data.csv"
if not DATA_PATH.exists():
	print(f"Dataset not found at {DATA_PATH}")
	sys.exit(1)

data = pd.read_csv(DATA_PATH)

X = data[["age", "weight", "duration"]]
y = data["calories"]


def train_and_save(output_path: Path):
	model = LinearRegression()
	model.fit(X, y)
	joblib.dump(model, str(output_path))


if __name__ == "__main__":
	out_path = Path(__file__).resolve().parent / "calorie_model.pkl"
	try:
		train_and_save(out_path)
	except Exception as exc:
		print(f"Training failed: {exc}")
		sys.exit(1)
	print(f"Model trained and saved to {out_path}")

