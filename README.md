# 🏠 HuurScout: Netherlands Rent Prediction

HuurScout is a machine learning-powered application that predicts rental prices for properties in the Netherlands. By analyzing both location-based and property-specific features, HuurScout provides users with insights into fair rental values, helping them make informed decisions in the real estate market.

---

## 📚 Table of Contents
- Overview
- Features
- Installation
- Usage
- Project Workflow
- Results
- Future Improvements
- Contributing
- License
- Acknowledgments

---

## 🖊 Overview
HuurScout predicts **rental prices** (Huurmaand Woning) in the Netherlands using:
- Cleaned and processed datasets with detailed property information.
- Machine learning models like **Random Forest Regressor**.
- Advanced feature engineering techniques such as **one-hot encoding**.
- An interactive **Streamlit dashboard** for real-time predictions.

---

## 🛠️ Features
Key attributes include:

### Property-Specific Attributes
- Numerical data: `oppervlakte_wonen`, `bouwjaar`, `slaapkamers`, `aantal_badkamers`.
- Categorical data: `energielabel`, `plaatsnaam`.

### Location Insights
- One-hot encoding of `plaatsnaam` enables accurate location-based predictions.

### Predictions
- Rental price estimates and deal classification based on user inputs.

---

## ⚙️ Installation
To set up the environment:

Clone the repository.

Navigate to the project folder.

Install dependencies using the `requirements.txt` file.

---

## 🚀 Usage
To run the application:

Launch the Streamlit app.

Enter details like `oppervlakte_wonen`, `bouwjaar`, and `energielabel`.

Select a property location from the `plaatsnaam` dropdown.

Click **Run Prediction** to view the rental price estimate and deal classification.

---

## 🧬 Project Workflow
### Data Processing
- Missing values are handled using imputation.
- Categorical features like `energielabel` and `plaatsnaam` are encoded.

### Feature Engineering
- Numerical and categorical data are combined for model training.
- Features are scaled to improve model performance.

### Model Training
- A **Random Forest Regressor** is used for predictions.
- Performance is evaluated using metrics like Mean Absolute Error (MAE) and R-squared (R²).

### Prediction
- Predictions are generated in real time via the Streamlit app.
- Deal classification is displayed to help users assess rental fairness.

---

## 📊 Results
The model achieves:
- High accuracy with an R² score of 0.89.
- Mean Absolute Error (MAE) of ~€150.
- Key predictors include `oppervlakte_wonen`, `plaatsnaam`, and `bouwjaar`.

---

## 🔮 Future Improvements
- Incorporate advanced models like XGBoost or LightGBM.
- Add geospatial analysis for better location-based insights.
- Include external data, such as public transport proximity and local amenities.

---

## 🤝 Contributing
Contributions are welcome. Fork the repository, create a branch for your feature, commit your changes, and open a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## 🙏 Acknowledgments
Thanks to the libraries and tools that made this project possible:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
