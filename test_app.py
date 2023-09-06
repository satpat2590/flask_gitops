import unittest
from app import app


class FlaskTestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
    
    def test_hello_world(self):
        # Get a response from the server, using it to test.
        response = self.app.get("/")
        # If response status code is 200, then OK 
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)
    
    def test_goodbye_world(self):
        # Get a response from the server, using it to test.
        response = self.app.get("/login")
        # If response status code is 200, then OK 
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<h1 class="title">AI Speakeasy</h1>', response.data)
    


if __name__=='__main__': 
    unittest.main()