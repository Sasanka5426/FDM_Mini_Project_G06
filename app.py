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

    if prediction == 0:
        prediction_text_val='No Re-admission'
    
    elif prediction == 1:
        prediction_text_val='After 30 days'
    
    elif prediction == 2:
        prediction_text_val='Before 30 days'
    
    
    return render_template('Classification.html', prediction_text=prediction_text_val.format(prediction))

    ####testing plot
    import plotly
    import plotly.graph_objs as go
    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
    # Create a trace
    data = [go.Scatter(
        x = df['data'],
        y = df['totale_positivi'],
    )]
    layout = go.Layout(
            xaxis=dict(
                title='Data',    
            ),
            yaxis=dict(
                title='Totale positivi',  
            )
        )
    fig = go.Figure(data=data, layout=layout)

    plotly.offline.plot(fig,filename='positives.html',config={'displayModeBar': False})

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