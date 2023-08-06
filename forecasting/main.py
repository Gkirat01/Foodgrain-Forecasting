from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima


app = Flask(__name__)

#the file which we want to read
dataset = pd.read_csv("C:\\Users\\intel\\Desktop\\project.csv", nrows=996)

#Dropping off the unnecessary columns
dataset= dataset.drop(['enquiry_status', 'terminal_id', 'rc_id', 'scheme_id', 'scheme_name', 'schemetype', 'balance_allotment', 'price_per_kg', 'allowed_allotment', 'type_of_trans','auth_mode', 'status', 'trans_status', 'member_name', 'amount', 'transaction_id','sale_fps_id','home_fps_id','created_on','updated_on','no_of_members','allocationmonth','allocationyear','commodity_code','balance_allotment','impds_trans_id','trans_id','session_id','uid','uid_token','receipt_id','uid_refer_no','remarks','date_time','member_id'], axis=1 )


#the month in the dataset is present in the form of integer type so lets convert it to its corresponding month name
import calendar
# Create a dictionary mapping the integer months to their corresponding names
month_mapping = {i: calendar.month_name[i] for i in range(1, 13)}
# Map the integer months to their corresponding names in the 'month' column
dataset['month'] = dataset['month'].map(month_mapping)

# Convert sale_state_code to state names
state_mapping = {
    27: 'Maharashtra',
    23: 'Madhya Pradesh',
    25: 'Meghalaya',
    3: 'Punjab',
    29: 'Karnataka',
    32: 'Kerala',
    7: 'Delhi',
    10: 'Bihar',
    16: 'Tripura',
    6: 'Haryana',
    21: 'Odisha',
    9: 'Uttar Pradesh',
    33: 'Tamil Nadu'
}
dataset['sale_state_code'] = dataset['sale_state_code'].map(state_mapping)

# Convert home_state_code to state names
home_state_mapping = {
    20: 'Jharkhand'
}
dataset['home_state_code'] = dataset['home_state_code'].map(home_state_mapping)


@app.route('/')
def index():
# Plotting the bar graph to show the relationship between sale state and home state
    state_relation = dataset.groupby(['sale_state_code', 'home_state_code']).size().reset_index(name='count')
    plt.figure(figsize=(12, 6))
    plt.bar(state_relation['sale_state_code'] + '-' + state_relation['home_state_code'], state_relation['count'],
            width=0.4)
    plt.xlabel('State Relationship (Sale State - Home State)')
    plt.ylabel('Count')
    plt.title('Relationship between Sale State and Home State')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig('C:\\Users\\intel\\PycharmProjects\\forecasting\\static\\plot.png')  # Save the plot to a file
    plt.close()

###############


    # Plotting the bar graph for total allotment in Delhi
    # Plotting the bar graph for total allotment in Delhi
    # Define the desired order of the months
    month_order = ["January", "February", "March", "April", "May", "June", "July"]

    # Filter the dataset for Delhi, Wheat, Rice, Pmgkay Wheat, and Pmgkay Rice and the months from January to July
    delhi_allotment = dataset[(dataset["sale_state_code"] == "Delhi") &
                          (dataset["commodity_name"].isin(["Wheat", "Rice", "Pmgkay Wheat", "Pmgkay Rice"])) &
                          (dataset["month"].isin(month_order))]

    # Convert the "month" column to categorical with the desired order
    delhi_allotment["month"] = pd.Categorical(delhi_allotment["month"], categories=month_order, ordered=True)

    # Group the data by month and calculate the total allotment for each month
    monthly_allotment = delhi_allotment.groupby("month")["total_allotment"].sum()

    # Get the total sum of total_allotment for each month
    total_sum_by_month = monthly_allotment.sum()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(monthly_allotment.index, monthly_allotment.values)
    plt.title("Total Distribution of Wheat, Rice, Pmgkay Wheat, and Pmgkay Rice in Delhi")
    plt.xlabel("Month")
    plt.ylabel("Total Distribution (in kg)")

    # Modify y-axis tick labels to display total sum of total_allotment
    y_labels = [f"{qty:.1f}K" for qty in monthly_allotment]
    plt.yticks(monthly_allotment, y_labels)


    # Print total sum of total_allotment on the top of each bar
    for bar in bars:
     height = bar.get_height()
     plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}K", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('C:\\Users\\intel\\PycharmProjects\\forecasting\\static\\line.png')  # Save the bar graph to a file
    plt.close()

################
    # Perform the ADF test on the monthly allotment data
    # Ho: It is non stationary
    # H1: It is stationary
    result = adfuller(monthly_allotment.values)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print(
            "Strong evidence against the null hypothesis (Ho). Reject the null hypothesis. Data has no unit root and is stationary.")
    else:
        print("Weak evidence against null hypothesis. Time series has a unit root, indicating it is non-stationary.")
###################

    # Perform differencing on the monthly allotment data
    diff_series = monthly_allotment.diff().dropna()
    # Perform the ADF test on the differenced series
    result = adfuller(diff_series.values)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print(
            "Strong evidence against the null hypothesis . Reject the null hypothesis. Differenced data has no unit root & is stationary.")
    else:
        print(
            "Weak evidence against null hypothesis. Differenced time series has a unit root, indicating it is non-stationary.")
#################

    # Fit the ARIMA model using auto_arima
    stepwise_fit = auto_arima(diff_series, trace=True, suppress_warnings=True)
    # Print the summary of the fitted model
    print(stepwise_fit.summary())
################
# split data into training and testing
    train_data = diff_series[:5]  # First 5 months for training
    test_data = diff_series[5:]  # Last month for testing
    print("Training Data:")
    print(train_data)
    print("\nTesting Data:")
    print(test_data)

##############
    # train the model
    import warnings
    warnings.filterwarnings("ignore")
    # Train the model
    model = ARIMA(train_data, order=(0, 0, 1))
    model_fit = model.fit()
    # Print the summary of the fitted model
    print(model_fit.summary())

################
    ##
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    # Predict using the trained model
    pred = model_fit.get_prediction(start=start, end=end, dynamic=False)
    # Extract the predicted values
    pred_values = pred.predicted_mean
    # Set the index of the predicted values
    pred_values.index = dataset.index[start: end + 1]
    print(pred_values)
##############
    # Print the predicted values and actual values
    for pred, actual in zip(pred_values, test_data):
        print("Predicted: {:.2f}  Actual: {:.2f}".format(pred, actual))
##############

    # Calculate MAE and RMSE
    mae = mean_absolute_error(test_data, pred_values)
    rmse = mean_squared_error(test_data, pred_values, squared=False)
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))

    #####
    accuracy = 100 - mae  # Calculate the accuracy as (100 - MAE)
    print("Model Accuracy: {:.2f}%".format(accuracy))
###############
    pred_text= "predicted value: {:.2f}".format(pred)
    actual_text= "actual value: {:.2f}".format(actual)
    mae_text = "Mean Absolute Error (MAE): {:.2f}".format(mae)
    rmse_text = "Root Mean Squared Error (RMSE): {:.2f}".format(rmse)
    accuracy_text = "Model Accuracy: {:.2f}%".format(accuracy)


###########

    forecast_steps = 3  # Adjust this value as needed
    # Get the forecasted values
    forecast = model_fit.get_forecast(steps=forecast_steps)
    # Extract the predicted mean values
    forecast_values = forecast.predicted_mean
    # Set the index for the forecasted values
    forecast_index = pd.date_range(start='2023-07-31', periods=forecast_steps + 1, freq='M')[1:]

    # Convert the forecasted values to the desired format (in kg)
    formatted_forecast = [f"{value:0.1f}K" for value in forecast_values]

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    bars = plt.bar(forecast_index, forecast_values, color='orange', label='Forecast')
    plt.xlabel('Month')
    plt.ylabel('Total Distribution (in kg)')
    plt.title('Forecasted Total Distribution')
    plt.legend()
    plt.xticks(rotation=45)

    # Print the forecasted values on top of the bars
    for bar, value in zip(bars, formatted_forecast):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, value, ha='center', va='bottom', rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig('C:\\Users\\intel\\PycharmProjects\\forecasting\\static\\predicted.png')  # Save the bar graph to a file
    plt.close()

    return render_template('index.html', plot_path='static/line.png',
                           new_path='static/plot.png',
                           actual_text=actual_text,
                           mae_text=mae_text,
                           rmse_text=rmse_text,
                           accuracy_text=accuracy_text,
                           pred_text=pred_text,
                           predicted_path='static/predicted.png')


if __name__ == '__main__':
    app.run()

