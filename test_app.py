import unittest
from app import app


class FlaskTestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
    
    def test_index(self):
        # Get a response from the server, using it to test.
        response = self.app.get("/", follow_redirects=True)
        # If response status code is 200, then OK 
        self.assertIn(b'<h1 class="title">AI Speakeasy</h1>', response.data)
    
    def test_login(self):
        # Provide sample username and password
        login = { 
            'username': 'testUser',
            'password': 'testPassword'
        }
        
        # POST the test login info to the Flask application, allowing you to enter index page
        self.app.post("/login", data=login, follow_redirects=True)

        # If response status code is 200, then OK 
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)
    


if __name__=='__main__': 
    unittest.main()