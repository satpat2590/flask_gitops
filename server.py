from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

app = FastAPI()


@app.get("/")
async def index():
    html_content = "<html><body><h1>Hello, World!</h1></body></html>"
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/greet/{name}")
async def greet(name: str):
    message = f"Hello, {name}!"
    html_content = f"<html><body><h1>{message}</h1></body></html>"
    return HTMLResponse(content=html_content, status_code=200)


from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Create an engine that connects to your database
engine = create_engine("sqlite:///people.db")

# Create a base class that models will inherit from
Base = declarative_base()


# Define a model as a subclass of the Base class
class Person(Base):
    __tablename__ = "person"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)


# Create tables
Base.metadata.create_all(engine)

# Create a Session class and use it to query the database
Session = sessionmaker(bind=engine)

with Session() as session:
    for person in session.query(Person).all():
        print(person.name, person.age)
