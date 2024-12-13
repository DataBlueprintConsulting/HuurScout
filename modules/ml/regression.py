import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np

def regression_analysis(df, new_data, actual_rent):
    """Perform regression analysis and make predictions based on user input."""
    if 'huurmaand_woning' not in df.columns:
        raise KeyError('The target column "huurmaand_woning" is not in the dataset.')

    # Start by selecting numeric features
    X = df.select_dtypes(include=['number'])

    # Add one-hot encoded categorical features
    one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
    X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')

    # Define the target variable
    y = df['huurmaand_woning']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = y.fillna(y.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)

    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')

    y_log = np.log(y)
    # Train model with y_log, predict, and then exponentiate predictions:
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Process new data for prediction
    new_data_filled = pd.DataFrame(columns=X.columns)
    for col in X.columns:
        new_data_filled[col] = [0]  # Default to 0 for all columns
    new_data_filled.update(new_data)

    new_data_imputed = pd.DataFrame(imputer.transform(new_data_filled), columns=X.columns)
    new_data_scaled = scaler.transform(new_data_imputed)
    predicted_rent = model.predict(new_data_scaled)[0]

    # Calculate deviation and classification
    deviation = (actual_rent - predicted_rent) / predicted_rent * 100
    classification = (
        "Great Deal" if deviation < -20 else
        "Good Deal" if deviation <= 10 else
        "Bad Deal"
    )

    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=deviation,
        number={'suffix': '%'},
        title={
            'text': f"HoeHardWordIkGenaaid-meter Deal Classification: {classification}<br>Predicted Rent: €{predicted_rent:,.2f} <br>Actual Rent: €{actual_rent:,.2f}"
        },
        gauge={
            'shape': 'angular',
            'axis': {
                'range': [-50, 50],
                'tickvals': [-50, -20, 0, 20, 50],
                'ticktext': ['-50%', '-20%', '0%', '+20%', '+50%']
            },
            'bar': {'thickness': 0},
            'steps': [
                {'range': [-50, 50], 'color': 'lightgray', 'thickness': 1},
                {'range': [-20, 0], 'color': 'palegreen', 'thickness': 1},
                {'range': [0, 20], 'color': 'orange', 'thickness': 1},
                {'range': [20, 50], 'color': 'red', 'thickness': 1},
                {'range': [-50, -20], 'color': 'green', 'thickness': 1},
                {'range': [0, deviation], 'color': 'blue', 'thickness': 0.8},
            ],
        }
    ))
    st.plotly_chart(fig)

    f = plt.figure(figsize=(12, 6))

    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Rent (Huurmaand Woning)')
    plt.ylabel('Predicted Rent')
    plt.title('Actual vs Predicted Rent')
    st.plotly_chart(f)

    return {
        "model": model,
        "mae": mae,
        "r2": r2,
        "predicted_rent": predicted_rent,  # Include predicted_rent in the return dictionary
        "Mean MAE from CV:": -scores.mean()
    }



# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
# import streamlit as st
# import plotly.graph_objects as go
# from sklearn.preprocessing import LabelEncoder

# def regression_analysis(df, new_data, actual_rent):
#     """Perform regression analysis and make predictions based on user input."""
#     if 'huurmaand_woning' not in df.columns:
#         raise KeyError('The target column "huurmaand_woning" is not in the dataset.')

#     # Start by selecting numeric features
#     X = df.select_dtypes(include=['number'])

#     # Add one-hot encoded categorical features
#     one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
#     X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')

#     # Define the target variable
#     y = df['huurmaand_woning']

#     # Handle missing values
#     imputer = SimpleImputer(strategy='mean')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#     y = y.fillna(y.mean())

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Scale the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train the model
#     model = RandomForestRegressor(random_state=42, n_estimators=100)
#     model.fit(X_train_scaled, y_train)

#     # Evaluate
#     y_pred = model.predict(X_test_scaled)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Process new data for prediction
#     new_data_filled = pd.DataFrame(columns=X.columns)
#     for col in X.columns:
#         new_data_filled[col] = [0]  # Default to 0 for all columns
#     new_data_filled.update(new_data)

#     new_data_imputed = pd.DataFrame(imputer.transform(new_data_filled), columns=X.columns)
#     new_data_scaled = scaler.transform(new_data_imputed)
#     predicted_rent = model.predict(new_data_scaled)[0]

#     # Calculate deviation and classification
#     deviation = (actual_rent - predicted_rent) / predicted_rent * 100
#     classification = (
#         "Great Deal" if deviation < -20 else
#         "Good Deal" if deviation <= 10 else
#         "Bad Deal"
#     )

#     # Create the gauge chart
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=deviation,
#         number={'suffix': '%'},
#         title={
#             'text': f"HoeHardWordIkGenaaid-meter Deal Classification: {classification}<br>Predicted Rent: €{predicted_rent:,.2f} <br>Actual Rent: €{actual_rent:,.2f}"
#         },
#         gauge={
#             'shape': 'angular',
#             'axis': {
#                 'range': [-50, 50],
#                 'tickvals': [-50, -20, 0, 20, 50],
#                 'ticktext': ['-50%', '-20%', '0%', '+20%', '+50%']
#             },
#             'bar': {'thickness': 0},
#             'steps': [
#                 {'range': [-50, 50], 'color': 'lightgray', 'thickness': 1},
#                 {'range': [-20, 0], 'color': 'palegreen', 'thickness': 1},
#                 {'range': [0, 20], 'color': 'orange', 'thickness': 1},
#                 {'range': [20, 50], 'color': 'red', 'thickness': 1},
#                 {'range': [-50, -20], 'color': 'green', 'thickness': 1},
#                 {'range': [0, deviation], 'color': 'blue', 'thickness': 0.8},
#             ],
#         }
#     ))
#     st.plotly_chart(fig)
#     return {
#         "model": model,
#         "mae": mae,
#         "r2": r2,
#         "predicted_rent": predicted_rent,  # Include predicted_rent in the return dictionary
#     }


# def regression_analysis(df, new_data, actual_rent):
#     """Perform regression analysis and make predictions based on user input."""
#     if 'huurmaand_woning' not in df.columns:
#         raise KeyError('The target column "huurmaand_woning" is not in the dataset.')

#     # Start by selecting numeric features
#     X = df.select_dtypes(include=['number'])

#     # Add one-hot encoded categorical features
#     one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
#     X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')

#     # Define the target variable
#     y = df['huurmaand_woning']

#     # Handle missing values
#     imputer = SimpleImputer(strategy='mean')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#     y = y.fillna(y.mean())

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Scale the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train the model
#     model = RandomForestRegressor(random_state=42, n_estimators=100)
#     model.fit(X_train_scaled, y_train)

#     # Evaluate
#     y_pred = model.predict(X_test_scaled)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Process new data for prediction
#     new_data_filled = pd.DataFrame(columns=X.columns)
#     for col in X.columns:
#         new_data_filled[col] = [0]  # Default to 0 for all columns
#     new_data_filled.update(new_data)

#     new_data_imputed = pd.DataFrame(imputer.transform(new_data_filled), columns=X.columns)
#     new_data_scaled = scaler.transform(new_data_imputed)
#     predicted_rent = model.predict(new_data_scaled)[0]

#     # Calculate deviation and classification
#     deviation = (actual_rent - predicted_rent) / predicted_rent * 100
#     classification = (
#         "Great Deal" if deviation < -20 else
#         "Good Deal" if deviation <= 10 else
#         "Bad Deal"
#     )


#     # Create the gauge chart
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=deviation, 
#         number={'suffix': '%'},
#         title={
#             'text': f"HoeHardWordIkGenaaid-meter Deal Classification: {classification}<br>Predicted Rent: €{predicted_rent:,.2f} <br>Actual Rent: €{actual_rent:,.2f}"
#         },
#         gauge={
#             'shape': 'angular',
#             'axis': {
#                 'range': [-50, 50],  
#                 'tickvals': [-50, -20, 0, 20, 50], 
#                 'ticktext': ['-50%', '-20%', '0%', '+20%', '+50%']
#             },
#             'bar': {'thickness': 0}, 
#             'steps': [
#                 {'range': [-50, 50], 'color': 'lightgray', 'thickness': 1},     # Background
#                 {'range': [-20, 0], 'color': 'palegreen', 'thickness': 1},      # Good Deal (negative)
#                 {'range': [0, 20], 'color': 'orange', 'thickness': 1},          # Slightly Bad Deal
#                 {'range': [20, 50], 'color': 'red', 'thickness': 1},            # Bad Deal
#                 {'range': [-50, -20], 'color': 'green', 'thickness': 1},        # Great Deal
#                 {'range': [0, deviation], 'color': 'blue', 'thickness': 0.8},   # Simulated needle
#             ],
#         }
#     ))
#     st.plotly_chart(fig)
    
#     f = plt.figure(figsize=(12, 6))

#     plt.scatter(y_test, y_pred, alpha=0.7)
#     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
#     plt.xlabel('Actual Rent (Huurmaand Woning)')
#     plt.ylabel('Predicted Rent')
#     plt.title('Actual vs Predicted Rent')
#     st.plotly_chart(f)

#     return {
#         "model": st.write('Model:', model),
#         "mae": st.write('Mae:', mae),
#         "r2": st.write('R^2:',r2),
#         "y_test": y_test,
#         "y_pred": y_pred
#     }
