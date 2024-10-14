from flask import Flask, request, send_file, render_template
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    # CRISP-DM Step 1: Business Understanding
    # In this step, we clarify the business objective of generating a linear regression model plot.
    
    try:
        # CRISP-DM Step 2: Data Understanding
        # Collect input parameters, understanding the user's requirements
        a = float(request.form['a'])  # Slope
        b = float(request.form['b'])  # Intercept
        noise = float(request.form['noise'])  # Noise level
        num_points = int(request.form['num_points'])  # Number of data points
    except KeyError as e:
        return {'error': f'Missing key: {str(e)}'}, 400

    # CRISP-DM Step 3: Data Preparation
    # Generate input data, using random number generation to simulate data points
    np.random.seed(0)  # Fix random seed for reproducibility
    x = np.random.randint(500, 1000, size=num_points)  # Random x values (independent variable)
    
    # Generate y values (dependent variable) with added noise
    y = a * x + b + np.random.normal(0, noise, size=num_points)  # Linear relationship with noise

    # CRISP-DM Step 4: Modeling
    # Fit the generated data using a linear regression model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    # CRISP-DM Step 5: Evaluation
    # Generate the plot to evaluate the model's performance
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Points')  # Data points
    plt.plot(x, model.predict(x.reshape(-1, 1)), color='red', label=f'Predicted Line: y = {a}x + {b}')  # Regression line
    plt.xlabel('x (Independent Variable)')
    plt.ylabel('y (Dependent Variable)')
    plt.title('Linear Regression Result')
    plt.legend()
    
    # Save the image
    image_path = 'static/linear_regression_plot.png'
    plt.savefig(image_path)
    plt.close()

    # CRISP-DM Step 6: Deployment
    # Return the generated plot for the user to view
    return send_file(image_path)

if __name__ == '__main__':
    # Create static folder to store images
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
