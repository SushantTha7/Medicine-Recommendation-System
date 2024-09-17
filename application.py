from flask import Flask, render_template, request, redirect, url_for, flash, session
# from flask_mysqldb import MySQL
import mysql.connector
from flask_bcrypt import Bcrypt
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from flask_mail import Mail, Message
from sklearn.tree import export_graphviz
import graphviz
import os


application = Flask(__name__)
# Set the secret key to a random string
application.secret_key = os.urandom(24)

model = pickle.load(open('model.pkl', 'rb'))
test = pd.read_csv("test_data.csv", error_bad_lines=False)
x_test = test.drop('prognosis', axis=1)
bcrypt = Bcrypt(application)

# # MySQL configuration
# application.config['MYSQL_HOST'] = 'localhost'
# application.config['MYSQL_USER'] = 'root'
# application.config['MYSQL_PASSWORD'] = 'mysql@123'
# application.config['MYSQL_DB'] = 'medicine'
# mysql = MySQL(application)

#establishing the connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="mysql@123",
    database="medicine"
)

# Create a cursor object
cur = conn.cursor()
# try:
#     cur = mysql.connection.cursor()
#     # Your SQL code here...
# except Exception as e:
#     print(f"Error connecting to MySQL: {e}")


# create table for history data

cur.execute('''
CREATE TABLE IF NOT EXISTS history_table (  
    id INT AUTO_INCREMENT PRIMARY KEY,
    symptoms VARCHAR(255),
    prediction VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES user_data(id) ON DELETE CASCADE
)
''')


#create table for user_data


cur.execute('''
CREATE TABLE IF NOT EXISTS user_data (
    id INT AUTO_INCREMENT PRIMARY KEY, 
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(255),
    address VARCHAR(255),
    password VARCHAR(255))
''')

#create table for contacts
cur.execute('''
CREATE TABLE IF NOT EXISTS contacts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    address VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(255),
    message TEXT)
''')

# Landing page route
@application.route('/')
def Landing_page():
    return render_template('index.html')

@application.route("/home")
def home():
    return render_template("home.html")

@application.route('/check')
def check_page():
    return render_template('letscheck.html')

@application.route('/about')
def about_page():
    return render_template('about.html')

@application.route('/team')
def team_page():
    return render_template('team.html')

@application.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('name', None)
    flash('You have been logged out', 'info')
    return render_template('index.html')

@application.route('/history')
def history_page():
    if 'user_id' not in session:
        flash("Please log in to use this feature.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur = conn.cursor()

    # Fetch data from the history table for the logged-in user
    cur.execute("SELECT symptoms, prediction, created_at FROM history_table WHERE user_id = %s", (user_id,))
    history = cur.fetchall()  # Fetch all records for the user
    cur.close()

    # Pass the fetched data to the template for rendering
    return render_template('history.html', data=history)

@application.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        password = request.form['password']

        # Form validation
        if not name:
            flash('Please enter your name')
            return redirect(request.url)
        elif not email:
            flash('Please enter your email')
            return redirect(request.url)
        elif not phone:
            flash('Please enter your phone number')
            return redirect(request.url)
        elif not address:
            flash('Please enter your address')
            return redirect(request.url)
        elif not password:
            flash('Please enter a password')
            return redirect(request.url)
        elif len(password) < 6:
            flash('Password must be at least 6 characters')
            return redirect(request.url)

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Insert data into user_data table
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO user_data (name, email, phone, address, password) VALUES (%s, %s, %s, %s, %s)',
            (name, email, phone, address, hashed_password)
        )
        conn.commit()
        cur.close()

        flash('Registered successfully')
        return redirect(url_for('login'))

    return render_template('register.html')


@application.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]  # Get the plain password input

        # Fetch user data from the database
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_data WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            # Check if the password is correct
            stored_password_hash = user[5]  # Assuming password hash is in the 5th column
            if bcrypt.check_password_hash(stored_password_hash, password):  # Use check_password_hash to validate
                session["user_id"] = user[0]  # Store user_id in session
                session["name"] = user[1]
                flash("Login successful!", "success")
                return redirect("/home")
            else:
                flash("Wrong password!", "danger")
        else:
            flash("Email not found!", "danger")

    return render_template("login.html")



@application.route('/contact', methods=['GET', 'POST'])
def contact_page():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']

        cur = conn.cursor()
        cur.execute("INSERT INTO contacts (name, address, email, phone, message) VALUES (%s, %s, %s, %s, %s)",
                    (name, address, email, phone, message))
        conn.commit()
        cur.close()

        # Send email (this assumes you have configured Flask-Mail)
        msg = Message('Medicine', sender='medicination1@gmail.com', recipients=[email])
        msg.html = render_template('mail.html')
        mail.send(msg)

        return render_template('ThankYou.html')

    return render_template('contact.html')

@application.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Check if the user is logged in
        if 'user_id' not in session:
            flash("Please log in to use this feature.")
            return redirect(url_for('login'))

        user_id = session['user_id']  # Retrieve user_id from session

        # Load test data and model (ensure they are properly initialized)
        test = pd.read_csv("test_data.csv")
        x_test = test.drop('prognosis', axis=1)
        y_test = test['prognosis']

        col = x_test.columns
        inputt = [str(x) for x in request.form.values()]

        # Check if all the input symptoms are among the 132 symptoms in the dataset
        if not all(symptom in col for symptom in inputt):
            return render_template('letscheck.html', pred="No symptoms found")

        b = [0] * 132
        for x in range(0, 132):
            for y in inputt:
                if col[x] == y:
                    b[x] = 1
        b = np.array(b)
        b = b.reshape(1, 132)
        prediction = model.predict(b)[0]

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, model.predict(x_test))

        # Store the prediction in the database
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO history_table (symptoms, prediction, user_id) VALUES (%s, %s, %s)',
            (str(inputt), prediction, user_id)
        )
        conn.commit()
        cur.close()

        return render_template('letscheck.html', symptoms=inputt, pred=f"The probable diagnosis says it could be {prediction}", accuracy=f"The accuracy score is {accuracy:.2f}")

@application.route('/decision-tree')
def decision_tree():
    dot_data = export_graphviz(model, out_file=None, feature_names=x_test.columns,
                               class_names=model.classes_, filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render('decision_tree')  # Save the image as 'decision_tree.png'
    return send_file("decision_tree.png", mimetype='image/png')

if __name__ == '__main__':
    application.run(debug=True, port=8001)