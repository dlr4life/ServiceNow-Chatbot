import pytest
from app import app, db
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client
        with app.app_context():
            db.drop_all()

def test_index_route(client):
    """Test the index route returns 200"""
    response = client.get('/')
    assert response.status_code == 200

def test_chat_endpoint(client):
    """Test the chat endpoint with a message"""
    data = {'message': 'Hello'}
    response = client.post('/chat',
                         data=json.dumps(data),
                         content_type='application/json')
    assert response.status_code == 200
    assert 'reply' in response.json

def test_settings_endpoint(client):
    """Test the settings endpoints"""
    # Test GET settings
    response = client.get('/api/settings')
    assert response.status_code == 302  # Redirect due to @login_required

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert 'status' in response.json
    assert 'components' in response.json

def test_metrics_endpoint(client):
    """Test the metrics endpoint"""
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'requests' in response.json
    assert 'cache' in response.json 