from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"

@app.route("/giggity")
def giggity():
    return """
        <h1>This is epic</h1>
        <ul>
            <li>This</li>
            <li>is</li>
            <li>epic</li>
            <li>, Giggity!</li>
        </ul>
    """


if __name__=='__main__': 
    app.run(debug=True, port=8001)