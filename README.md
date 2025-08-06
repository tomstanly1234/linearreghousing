# linearreghousing
Building A Linear Regression curve from scratch using multiple parameters and then using correlation and removing parameters which are correlated to one another(more than 0.85)....
# 🏠 California Housing Price Prediction (Linear Regression from Scratch)

This project implements a basic **Linear Regression** model from scratch using only **NumPy** to predict California housing prices. It demonstrates the full pipeline of:

- Data preprocessing
- Feature encoding and scaling
- Matrix-based gradient descent
- Loss minimization
- Model interpretation

---

## 📂 Files

- `housingdata.py` – Main script that:
  - Loads the dataset (`housing.csv`)
  - Handles missing data
  - Encodes categorical columns (`ocean_proximity`)
  - Scales features using `StandardScaler`
  - Implements gradient descent to learn weights
  - Outputs final model coefficients

---

## 📊 Features Used

- `latitude`
- `total_bedrooms`
- `median_income`
- `ocean_proximity` (label encoded)

---

## 🚀 How to Run

1. Make sure `housing.csv` (California housing dataset) is in the same folder.
2. Run the script:
   ```bash
   python housingdata.py
