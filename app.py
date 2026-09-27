
import streamlit as st
import joblib
import pandas as pd

# Load the trained models
linear_model = joblib.load('linear.sav')
logistic_model = joblib.load('logistic.sav')

st.title('ABC Ltd - Customer Churn & Spend Predictor')
st.write('Enter what you know about a customer from your sales records to predict '
         'whether they will stop buying in the next 90 days and how much they will spend.')

# Input fields
recency = st.slider('Days since last order', 0, 365, 60)
orders = st.slider('Number of orders in the last 12 months', 1, 200, 3)
spend = st.number_input('Total spend in the last 12 months (£)', 0.0, 350000.0, 1000.0, step=100.0)
tenure = st.slider('Days since first order', 0, 640, 300)
products = st.slider('Different products bought in the last 12 months', 1, 1000, 40)
returns = st.slider('Returns as % of spend', 0, 100, 0)

# Create a button to make predictions
if st.button('Predict'):
    # Prepare the input data in the same column order used for training
    input_data = pd.DataFrame(
        [[recency, orders, spend, tenure, products, returns / 100]],
        columns=['recency_days', 'orders_12m', 'spend_12m', 'tenure_days',
                 'unique_products_12m', 'return_rate'])

    # Make predictions
    churn = logistic_model.predict(input_data)[0]
    churn_prob = logistic_model.predict_proba(input_data)[0][1]
    spend_pred = max(linear_model.predict(input_data)[0], 0)
    revenue_at_risk = churn_prob * spend * 90 / 365

    if churn == 1:
        st.error(f'Likely to CHURN - probability {churn_prob:.0%}')
    else:
        st.success(f'Likely to STAY - probability of churn {churn_prob:.0%}')

    st.info(f'Predicted spend in the next 90 days: £{spend_pred:,.2f}')
    st.warning(f'Revenue at risk in the next 90 days: £{revenue_at_risk:,.2f}')

    if churn_prob >= 0.7:
        st.write('**Suggested action:** Call this customer this week with a win-back offer.')
    elif churn_prob >= 0.4:
        st.write('**Suggested action:** Account manager check-in / add to nurture email campaign.')
    else:
        st.write('**Suggested action:** Business as usual.')
