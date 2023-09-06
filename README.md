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

## Github Actions Workflow

The Github Actions workflow is interesting. It generates a job on a push or pull_request. 

This job will run an ubuntu environment, and then use a pre-existing workflow to download a python (version 3.9) environment. Then, it will begin to download the dependencies and then run the test file to make sure it works properly.