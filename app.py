from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_transtornos_sueño.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        sleep_duration = float(request.form['sleep_duration'])
        physical_activity = float(request.form['physical_activity'])
        bmi_category = float(request.form['bmi_category'])
        blood_pressure = float(request.form['blood_pressure'])

        # Crear un DataFrame con todas las características necesarias
        data = {
            'Age': [0],  # Valor predeterminado o promedio
            'Occupation': [0],  # Valor predeterminado o promedio
            'Sleep Duration': [sleep_duration],
            'Quality of Sleep': [0],  # Valor predeterminado o promedio
            'Physical Activity Level': [physical_activity],
            'Stress Level': [0],  # Valor predeterminado o promedio
            'BMI Category': [bmi_category],
            'Blood Pressure': [blood_pressure],
            'Heart Rate': [0],  # Valor predeterminado o promedio
            'Daily Steps': [0],  # Valor predeterminado o promedio
            'Gender_Female': [0],  # Valor predeterminado o promedio
            'Gender_Male': [0]  # Valor predeterminado o promedio
        }

        input_data = pd.DataFrame(data)
        app.logger.debug(f'DataFrame de entrada creado: {input_data}')

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [2, 4, 6, 7]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)
        app.logger.debug(f'Predicción: {prediccion[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediccion[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

