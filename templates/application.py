from os import name
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import pickle
from flask_mail import Mail, Message
from flask import Flask, render_template, request, redirect, url_for, flash,session
from flask_mysqldb import MySQL
import bcrypt

application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

test = pd.read_csv("test_data.csv", error_bad_lines=False)
x_test = test.drop('prognosis', axis=1)

application.config['MYSQL_HOST'] = 'localhost'
application.config['MYSQL_USER'] = 'root'
application.config['MYSQL_PASSWORD'] = 'mysql@123'
application.config['MYSQL_DB'] = 'medicine' #dbname


mysql = MySQL(application)


# create table for history data
mysql.connection.cursor().execute('''
CREATE TABLE IF NOT EXISTS history_table (  
    id INT AUTO_INCREMENT PRIMARY KEY,
    symptoms VARCHAR(255),
    prediction VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

@application.route('/')
def Landing_page():
    return render_template('index.html')


@application.route('/check')
def check_page():
    return render_template('letscheck.html')

@application.route('/logout')
def logout():
    return render_template('index.html')


@application.route('/about')
def about_page():
    return redirect(url_for('login'))


@application.route('/team')
def team_page():
    return render_template('team.html')


@application.route('/contact')
def contact_page():
    return render_template('contact.html')


@application.route('/history')
def history_page():
    return render_template('history.html')


@application.route('/send', methods=['GET', 'POST'])
def send():


@application.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # get form data
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        password = request.form['password']

        # form validation
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

        # insert data into user_data table
        cur = mysql.connection.cursor()
        cur.execute(
            'INSERT INTO user_data (name, email, phone, address, password) VALUES (%s, %s, %s, %s, %s)',
            (name, email, phone, address, password)
        )
        mysql.connection.commit()
        cur.close()

        flash('Registered successfully')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login Route


@application.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"].encode("utf-8")

        # Fetch user data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM user_data WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            # Check if the password is correct
            if bcrypt.checkpw(password, user[4].encode("utf-8")):
                session["user_id"] = user[0]
                session["name"] = user[1]
                flash("Login successful!", "success")
                return redirect("/")
            else:
                flash("Wrong password!", "danger")
        else:
            flash("Email not found!", "danger")

    return render_template("login.html")


# Logout Route
@application.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@application.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'user_id' not in session:
            flash("Please log in to use this feature.")
            return redirect(url_for('login'))

        user_id = session['user_id']  # Retrieve user_id from session

        # Load test data and model
        test = pd.read_csv("test_data.csv")
        x_test = test.drop('prognosis', axis=1)
        y_test = test['prognosis']

        col = x_test.columns
        inputt = request.form.getlist('choice[]')  # Retrieve all selected symptoms

        # Check if all the input symptoms are valid
        if not all(symptom in col for symptom in inputt):
            return render_template('letscheck.html', pred="No symptoms found")

        b = [0] * 132
        for x in range(0, 132):
            for y in inputt:
                if col[x] == y:
                    b[x] = 1
        b = np.array(b).reshape(1, 132)
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


#Display 
@application.route('/history')
def history():

    # Fetch all the data from the prediction_history table
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM history_table")
    data = cur.fetchall()
    cur.close()

    # Render a new HTML page to display the data in a tabular form
    return render_template('history.html', data=data)


if __name__ == "__main__":
    application.run(debug=True, port=8001)
