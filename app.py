import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

import shutil
import atexit

app = Flask(__name__)
app.secret_key = "secret"  # Change this to your desired secret key
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_uploads_folder():
    # Clean up the uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error cleaning up file or directory: {file_path} - {e}")

def cleanup_static_folder(exception=None):
    # Clean up the static folder, excluding main.css
    static_folder = app.config['STATIC_FOLDER']
    for filename in os.listdir(static_folder):
        if filename != 'main.css':  # Exclude main.css
            file_path = os.path.join(static_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error cleaning up file or directory: {file_path} - {e}")

# Register a function to be called when the application context is torn down
def cleanup_folders(exception=None):
    cleanup_uploads_folder()
    cleanup_static_folder()

# Register a function to be called when the Python interpreter exits
atexit.register(cleanup_folders)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect(url_for('analyze', filename=filename))
        else:
            flash('Allowed file types are xlsx, xls')
            return redirect(request.url)

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(filepath)

    # Plotting scatter plot
    plt.figure(figsize=(10, 6))
    colors = {0: 'orange', 1: 'blue'}
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2].map(colors))
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('Scatter Plot showing different '+ df.columns[2])
    plt.legend()
    plot_filename = f'plot_{filename.split(".")[0]}.png'
    plot_filepath = os.path.join('static', plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    # Data Profiling
    profile = ProfileReport(df)
    profile_filename = f'profile_{filename.split(".")[0]}.html'
    profile_filepath = os.path.join('static', profile_filename)
    profile.to_file(profile_filepath)

    return render_template('analyse.html', plot=plot_filename, profile=profile_filename, filename=filename)

@app.route('/train/<filename>', methods=['GET', 'POST'])
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(filepath)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Train SVM with RBF kernel
    svm_model = SVC(kernel='rbf', gamma='scale')
    svm_model.fit(X_train_scaled, y_train)

    # Evaluate models
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred)

    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred)
    
    def plot_decision_boundary(model, X, y, filename, x_coord=None, y_coord=None):
        plt.figure(figsize=(10, 6))
        cmap_light = ListedColormap(['orange', 'blue'])
        h = .02
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        colors = {0: 'orange', 1: 'blue'}
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.map(colors), cmap=ListedColormap(['orange', 'blue']), edgecolors='k')
        if x_coord is not None and y_coord is not None:
            plt.scatter([x_coord], [y_coord], c='red', s=100, edgecolors='k')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title('Decision Boundary')

        plot_filename = f'plot_{filename}'
        plot_filepath = os.path.join('static', plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        return plot_filename

    x_coord = None
    y_coord = None
    lr_prediction = None
    svm_prediction = None
    prediction = False

    if request.method == 'POST':
        x_coord = float(request.form['x_coord'])
        y_coord = float(request.form['y_coord'])
        new_data = scaler.transform([[x_coord, y_coord]])
        lr_prediction = lr_model.predict(new_data)[0]
        svm_prediction = svm_model.predict(new_data)[0]
        prediction = True

    # Plot lr decision boundary
    lr_plot_filename = plot_decision_boundary(lr_model, X, y, f'lr_boundary_{filename.split(".")[0]}.png', x_coord, y_coord)

    # Plot svm decision boundary
    svm_plot_filename = plot_decision_boundary(svm_model, X, y, f'svm_boundary_{filename.split(".")[0]}.png', x_coord, y_coord)

    return render_template('predict.html', plot=lr_plot_filename, plot2=svm_plot_filename, lr_accuracy=lr_accuracy, svm_accuracy=svm_accuracy, lr_report=lr_report, svm_report=svm_report, filename=filename, prediction=prediction, x_coord=x_coord, y_coord=y_coord, lr_prediction=lr_prediction, svm_prediction=svm_prediction)

if __name__ == '__main__':
    app.run(debug=True)
