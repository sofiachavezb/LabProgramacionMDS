
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from optimize import get_best_model

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo optimizado
model = get_best_model()


# Inicializar la aplicación Flask y la API con Swagger
app = Flask(__name__)
api = Api(app, version='1.0', title='Potabilidad de Agua API',
          description='Una API para predecir la potabilidad del agua basándose en características químicas.')

# Definir el modelo de datos (esquema) para la solicitud de predicción
potabilidad_model = api.model('PotabilidadModel', {
    'ph': fields.Float(required=True, description='Valor de pH del agua'),
    'Hardness': fields.Float(required=True, description='Dureza del agua'),
    'Solids': fields.Float(required=True, description='Sólidos disueltos en el agua'),
    'Chloramines': fields.Float(required=True, description='Concentración de cloraminas'),
    'Sulfate': fields.Float(required=True, description='Concentración de sulfatos'),
    'Conductivity': fields.Float(required=True, description='Conductividad del agua'),
    'Organic_carbon': fields.Float(required=True, description='Carbono orgánico'),
    'Trihalomethanes': fields.Float(required=True, description='Concentración de trihalometanos'),
    'Turbidity': fields.Float(required=True, description='Turbidez del agua')
})

# Endpoint principal para describir la API
@api.route('/')
class Home(Resource):
    def get(self):
        return {
            'message': 'Esta es la API para predecir la potabilidad del agua.'
        }

# Endpoint para realizar la predicción de potabilidad
@api.route('/potabilidad/')
class Potabilidad(Resource):
    @api.expect(potabilidad_model)
    def post(self):
        # Obtener los datos enviados por el usuario
        data = request.json

        # Extraer las características
        features = [
            data['ph'], data['Hardness'], data['Solids'],
            data['Chloramines'], data['Sulfate'], data['Conductivity'],
            data['Organic_carbon'], data['Trihalomethanes'], data['Turbidity']
        ]
        
        # Realizar la predicción usando el modelo cargado
        prediction = model.predict([features])[0]
        
        # Devolver la predicción como JSON
        return jsonify({'potabilidad': int(prediction)})

if __name__ == '__main__':
    # Ejecutar el servidor en el puerto 5001
    app.run(debug=True, port=5001)
