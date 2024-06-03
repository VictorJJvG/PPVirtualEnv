This app was created by Victor van Geest 08-01-2024

1) (optional) setup a virtual environment (and activate it)
2) install dependencies with 'pip install -r requirements.txt'
3) create a file called 'uploads' in your repository 
4) Run the app with 'python app.py'
5) localhost:5000 can now be searched in a browser to display the web application
6) Excel files exported as utf-8 can be uploaded from a PC to automatically be analysed 
    - The Excel file needs to have 3 columns, all with titles
    - The first and second column may contain any real numbers
    - The third column should either be a label 0 or 1

Example files:
In the folder pp_exampledata you can find 3 example excel files to upload to the webapp.
    - pp_beepboopbaap.xlsx shows quite random data on a plot, of course following the excel rules as described in 6) above. After uploading the user can analyse and predict what X (beep) and Y (boop) coordinates generate a certain Z (baap) label of 1 or 0.
    - pp_leanthagesex.xlsx allows a user to analyse with the X = length and Y = age of babies whether they are most likely male = 1 or female = 0.
    - pp_ThicnessClayChloritic.xlsx allows a user to analyse with the X = Thicness and Y = ClayPercentage of drillholes in Chinchina whether they are most likely to be on Chloritic soil = 1 or not = 0.
