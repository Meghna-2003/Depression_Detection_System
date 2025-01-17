from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
from quiz_module import predict_depression_level
from facial_detection import detect_facial_depression

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Load dataset columns for quiz
df = pd.read_csv('your_file.csv')
quiz_questions = [col for col in df.columns if col != 'depression']

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('quiz.html', questions=quiz_questions)

@app.route('/quiz', methods=['POST'])
def quiz():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Collect user inputs
    user_inputs = [float(request.form.get(q)) for q in quiz_questions]
    
    # Predict depression level
    depression_level = predict_depression_level(user_inputs)
    
    # Convert to standard Python int before saving to the session
    session['depression_level'] = int(depression_level)
    
    return render_template('result.html', level=session['depression_level'])


@app.route('/facial_detection', methods=['GET', 'POST'])
def facial_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        facial_depression_level = detect_facial_depression()
        return render_template('result.html', level=facial_depression_level)
    return render_template('facial_detection.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
