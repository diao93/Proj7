# Library imports
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import shap
import uvicorn

# Create a FastAPI instance
app = FastAPI()

# Load model and data
model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv('test_df_sample.csv')
data_train = pd.read_csv('train_df_sample.csv')

# Preprocessing
cols = data.select_dtypes(['float64']).columns
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])

cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])

explainer = shap.TreeExplainer(model['classifier'])


# Routes
@app.get('/')
def welcome():
    return {'message': 'Welcome to the API'}

@app.get('/{client_id}')
def check_client_id(client_id: int):
    exists = client_id in list(data['SK_ID_CURR'])
    return {'exists': exists}

@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    client_data = data[data['SK_ID_CURR'] == client_id]
    if client_data.empty:
        return JSONResponse(content={'error': 'Client ID not found'}, status_code=404)

    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.predict_proba(info_client)[0][1]
    return {'probability': round(float(prediction), 6)}

@app.get('/clients_similaires/{client_id}')
def get_data_voisins(client_id: int):
    features = list(data_train_scaled.columns)
    if 'SK_ID_CURR' in features: features.remove('SK_ID_CURR')
    if 'TARGET' in features: features.remove('TARGET')

    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn.fit(data_train_scaled[features])

    reference_observation = data_scaled[data_scaled['SK_ID_CURR'] == client_id][features].values
    if reference_observation.size == 0:
        return JSONResponse(content={'error': 'Client ID not found'}, status_code=404)

    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = data_train.iloc[indices[0], :]

    return df_voisins.to_dict(orient='records')

@app.get('/shaplocal/{client_id}')
def shap_values_local(client_id: int):
    client_data = data_scaled[data_scaled['SK_ID_CURR'] == client_id]
    if client_data.empty:
        return JSONResponse(content={'error': 'Client ID not found'}, status_code=404)

    client_data = client_data.drop('SK_ID_CURR', axis=1)
    shap_val = explainer(client_data)[0][:, 1]

    return {
        'shap_values': shap_val.values.tolist(),
        'base_value': shap_val.base_values,
        'data': client_data.values.tolist(),
        'feature_names': client_data.columns.tolist()
    }

@app.get('/shap/')
def shap_values():
    data_no_id = data_scaled.drop('SK_ID_CURR', axis=1)
    shap_val = explainer.shap_values(data_no_id)

    return {
        'shap_values_0': shap_val[0].tolist(),
        'shap_values_1': shap_val[1].tolist()
    }

# Run server
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
