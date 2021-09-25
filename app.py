#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#default page of our web-app
@app.route('/')
def home():
    return render_template('Classification.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    print(request.form.values())
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]

    #############################

    #############################

    prediction = model.predict(final_features)
    #output = round(prediction[0] ) 
    return render_template('Classification.html', prediction_text=' Customer Segmentation is :{}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)












#from flask import Flask, request, jsonify, render_template
# import pickle
# import pandas as pd
# import numpy as np


# app = Flask(__name__, template_folder='templates')




# #########################################################################################################
# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')

# def home():

#     return render_template('Classification.html')
# #methods=['POST']
# @app.route('/predict')
# def predict():

#     # int_features = [int(x) for x in request.form.values()]
#     # final_features = [np.array(int_features)]
#     #prediction = model.predict(final_features)

#     #print("zdvnklsdjbnv")

#     #output = round(prediction[0], 2)

#     #return render_template('Classification.html', prediction_text='Sales should be $ {}'.format(output))
#     return "Hello"



# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)