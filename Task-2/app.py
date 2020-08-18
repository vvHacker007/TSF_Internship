from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Regressor_task2_model.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
     if request.method == 'POST':
            
        Hr = float(request.form['Hours'])
        
        data = np.array(Hr)
        data= data.reshape(1,-1)
        my_prediction = model.predict(data)
        output = round(my_prediction[0],2)
        return render_template('result.html', prediction_text = output)
        
            

if  __name__ == '__main__':
    app.run(debug = True)