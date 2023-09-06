from flask import Flask, flash, session, request, redirect, url_for, render_template
from markupsafe import escape

app = Flask(__name__)
app.secret_key = 'superkey'

username = "user"
password = "pass"

@app.route("/")
def index():
    global username, password
    print(username, password, "index testing")
    if 'username' in session: 
        return "<h1>Hello, World!</h1>"
    else:
        return redirect(url_for('login'))
    

@app.route("/login", methods=["GET", "POST"])
def login(): 
    global username, password

    if request.method == 'GET':
        return render_template('login.html')
    
    print(request.form['password'], request.form['username'])
    if len(request.form['password']) <= 5:
        flash("Password must have more than 5 characters")
        return redirect(url_for('login'))
 
    username = request.form['username']
    password = request.form['password']
    session['username'] = username
    return redirect(url_for('index'))


@app.route("/<name>")
def hello_name(name):
    return f"<h1>Hello, {escape(name)}!</h1>"



if __name__=='__main__': 
    app.run(debug=True, port=8001)