# flask_gitops
Learning how to use Github Actions within this small Flask application which I'll (hopefully) scale up soon!

## Flask Application

Welcome to this intro to Flask and GitOps tools like Github Actions! 

Our directory contains the app.py file which is a starter template for a Flask application running on port 8001.

Eventually, this will become a custom application which uses the OpenAI module to simulate a conversation with an AI model. Overtime, this will become a template for my other AI applications which require small tweaks to the overall agent. 

Running the application is simple: 

```bash
python main.py || python3 main.py
```

Want to test it before? 
```bash
python test_app.py || python3 test_app.py
```

## GitHub Actions Workflow

The Github Actions workflow is interesting. It generates a job on a push or pull_request. 

This job will run an ubuntu environment, and then use a pre-existing workflow to download a python (version 3.9) environment. Then, it will begin to download the dependencies and then run the test file to make sure it works properly.

## Set Up for Embeddings

If you made it here, it means you're probably trying to embed your own PDFs or other files so that the models can accurately reference them. Let's get you started so that you 
can turn PDFs into well-formatted text. 

### For Machines Using 'apt' Package Manager

Go into your terminal and run the following: 

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

This should install Google's Tesseract OCR (Optical Character Recognition) within your machine. With this installed, the python module (pytesseract) should be able to run 
from flask_gitops/embeddings.py and convert PDFs -> Images -> Text 

### For MacOS Using 'brew' Package Manager

Go into your terminal and run the following: 

```bash
brew update --auto-update
brew install tesseract
```