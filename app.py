from flask import Flask, render_template, request
import sweetviz as sv
import pandas as pd
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

def profile_csv(df):
    try:
        # Make column names lowercase to handle case sensitivity
        df.columns = map(str.lower, df.columns)

        # Generate Sweetviz profile report
        report = sv.analyze(df)

        # Save the report to HTML
        report_html_path = 'report.html'
        report.show_html(filepath=report_html_path, open_browser=False)

        # Read the saved HTML report
        with open(report_html_path, 'r') as file:
            html_profile = file.read()

        return html_profile

    except Exception as e:
        return f"Error profiling data: {e}"

def plot_scatter(df):
    # Create a scatter plot using matplotlib
    fig, ax = plt.subplots()

    # Convert the index to a list before performing boolean indexing
    index_list = df.index.tolist()

    # Plot points where the third column is 1
    ax.scatter(df.iloc[[i for i in index_list if df.iloc[i, 2] == 1], 0], 
               df.iloc[[i for i in index_list if df.iloc[i, 2] == 1], 1], 
               marker='o', color='b', label='Value 1')

    # Plot points where the third column is 0
    ax.scatter(df.iloc[[i for i in index_list if df.iloc[i, 2] == 0], 0], 
               df.iloc[[i for i in index_list if df.iloc[i, 2] == 0], 1], 
               marker='x', color='r', label='Value 0')

    ax.set_title(df.columns[2])
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend()

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Encode the image to base64 for embedding in HTML
    plot_encoded = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()

    return f'<img src="data:image/png;base64,{plot_encoded}">'

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    html_table = ''  # Initialize variable to hold HTML table content
    html_profile = ''  # Initialize variable to hold HTML profile report content
    plot_encoded = ''  # Initialize variable to hold base64-encoded plot

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            try:
                # Save the uploaded file to the 'uploads' folder
                filename = secure_filename(uploaded_file.filename)
                save_path = 'uploads/' + filename
                uploaded_file.save(save_path)

                # Read the uploaded CSV file for rendering the DataFrame as an HTML table
                df = pd.read_csv(save_path, dtype=float, sep=';', decimal=',', encoding='utf-8')
                html_table = df.head(100).to_html(classes='table table-striped')

                # Generate and encode the scatter plot
                plot_encoded = plot_scatter(df)

                # Profile the uploaded CSV file
                html_profile = profile_csv(df)

            except Exception as e:
                return f"Error processing file: {e}"

    return render_template('analyse.html', table=html_table, profile=html_profile, plot_encoded=plot_encoded)

# Define a function to perform file cleanup
def cleanup_uploaded_files(exception=None):
    uploads_folder = 'uploads'
    for filename in os.listdir(uploads_folder):
        file_path = os.path.join(uploads_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Register the cleanup function to run when the app context is torn down
@app.teardown_appcontext
def teardown_appcontext(exception=None):
    cleanup_uploaded_files(exception)

if __name__ == '__main__':
    app.run(debug=True)
