import os  # Import the os module
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which is thread-safe
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template, request
from threading import Lock

app = Flask(__name__)
plot_lock = Lock()

def generate_data(a, num_points, noise):
    """
    CRISP-DM Step 3: Data Preparation
    Generates linear data with specified slope, number of points, and noise.

    Parameters:
    - a (float): Slope of the true line.
    - num_points (int): Number of data points to generate.
    - noise (float): Standard deviation of the Gaussian noise.

    Returns:
    - x (np.ndarray): Independent variable values.
    - y (np.ndarray): Dependent variable values with noise.
    """
    b = np.random.uniform(-10, 10)  # Random intercept between -10 and 10
    x = np.linspace(0, 10, num_points)
    y = a * x + b + np.random.normal(0, noise, num_points) 
    return x, y, b  # Return b as well

def perform_regression(x, y):
    """
    CRISP-DM Step 4: Modeling
    Performs linear regression on the given data.

    Parameters:
    - x (np.ndarray): Independent variable values.
    - y (np.ndarray): Dependent variable values.

    Returns:
    - slope (float): Estimated slope from regression.
    - intercept (float): Estimated intercept from regression.
    - mse (float): Mean Squared Error of the model.
    - r2 (float): R-squared score of the model.
    - y_pred (np.ndarray): Predicted y values from the model.
    """
    model = LinearRegression(fit_intercept=True)  # Ensure intercept is estimated
    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model.coef_[0], model.intercept_, mse, r2, y_pred

def create_plot(x, y, y_pred, a, b):
    """
    CRISP-DM Step 5: Evaluation
    Creates a plot of the data points, regression line, and true line.

    Parameters:
    - x (np.ndarray): Independent variable values.
    - y (np.ndarray): Dependent variable values.
    - y_pred (np.ndarray): Predicted y values from regression.
    - a (float): True slope used in data generation.
    - b (float): True intercept used in data generation.

    Returns:
    - plot_data (str): Base64-encoded PNG image of the plot.
    """
    with plot_lock:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Data Points', alpha=0.6, s=10)  # Smaller points for clarity
        plt.plot(x, y_pred, color='red', label='Regression Line', linewidth=2)
        plt.plot(x, a * x + b, color='green', linestyle='--', label=f'True Line (y = {a}x + {b})', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Result')
        plt.legend()
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return plot_data

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    CRISP-DM Steps:
    - Step 1: Business Understanding
    - Step 2: Data Understanding
    - Steps 3-6 are handled within helper functions and this route.
    """
    if request.method == 'POST':
        try:
            # Step 2: Data Understanding
            a = float(request.form['a'])           # Slope
            noise = float(request.form['noise'])   # Noise level
            num_points = int(request.form['num_points'])  # Number of data points

            # Input validation
            if num_points <= 0:
                return render_template('index.html', error='Number of points must be a positive integer.')
            if noise < 0:
                return render_template('index.html', error='Noise level must be non-negative.')
        except KeyError as e:
            return render_template('index.html', error=f'Missing parameter: {str(e)}')
        except ValueError:
            return render_template('index.html', error='Invalid input. Please ensure all inputs are numeric.')

        # Step 3: Data Preparation
        x, y, b = generate_data(a, num_points, noise)

        # Step 4: Modeling
        slope, intercept, mse, r2, y_pred = perform_regression(x, y)

        # Step 5: Evaluation
        plot_data = create_plot(x, y, y_pred, a, b)

        # Step 6: Deployment
        return render_template('index.html', plot=plot_data)

    # For GET requests, simply render the form
    return render_template('index.html')

if __name__ == '__main__':
    # CRISP-DM Step 6: Deployment
    # Ensure the 'static' and 'templates' directories exist
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, threaded=True)
