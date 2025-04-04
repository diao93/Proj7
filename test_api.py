from fastapi import status
import requests
import json

#API_URL = "http://127.0.0.1:8000/"
API_URL = "https://jngjhhdkqnkcvh2hyfjgmp.streamlit.app/"


def test_welcome():
    """Test la fonction welcome() de l'API."""
    response = requests.get(API_URL)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 'Welcome to the API'


def test_check_client_id():
    """Test la fonction check_client_id() de l'API avec un client faisant partie de la base de données."""
    url = API_URL + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == True


def test_check_client_id_2():
    """Test la fonction check_client_id() de l'API avec un client ne faisant pas partie de la base de données."""
    url = API_URL + str(100000)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == False


def test_get_prediction():
    """Test la fonction get_prediction() de l'API."""
    url = API_URL + "prediction/" + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 0.4805479971101088
