#import libraries
import numpy as np
import pandas as pd
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
    try:
        print(request.form.values())
        int_features = [x for x in request.form.values()]
        final_features = [np.array(int_features)]


        dataset = pd.read_csv("Data//demo_dataset.csv")
        id = int(final_features[0])
        print(type(id))
    

        #############################
        

        #dataset = pd.read_csv("demo_dataset.csv")

        #id = 430751654  After 30 days

        dataset = dataset.loc[dataset['encounter_id'] == id]
        encounterID = dataset['encounter_id'].values
        patient_nbr = dataset['patient_nbr'].values
        payer_code = dataset['payer_code'].values

        # dropping unwanted columns
        dataset = dataset.drop(['encounter_id','patient_nbr','weight','payer_code','medical_specialty','diag_1','diag_2','diag_3'],axis=1)
        

        # replacing values
        dataset["race"].replace({"?":"Unknown"}, inplace=True)

        # removing rows that contains 'Unknown/Invalid' for gender column
        dataset = dataset[dataset.gender != 'Unknown/Invalid']

        # data encoding

        dataset['race'] = dataset['race'].map({'Caucasian':0, 'AfricanAmerican':1, 'Asian':2, 'Hispanic':3, 'Other':4, 'Unknown':5})
        dataset['gender'] = dataset['gender'].map({'Male':1,'Female':0})
        dataset['age'] = dataset['age'].map({'[0-10)':1,'[10-20)':2, '[20-30)':3, '[30-40)':4, '[40-50)':5, '[50-60)':6, 
                                            '[60-70)':7, '[70-80)':8, '[80-90)':9, '[90-100)':10})

        # data encoding (col 16-38)

        for col in dataset.iloc[:,16:39]:
            dataset[col] = dataset[col].map({'No':0, 'Steady':1, 'Up':2, 'Down':3})

        # data encoding
        dataset['change'] = dataset['change'].map({'No':0, 'Ch':1})
        dataset['diabetesMed'] = dataset['diabetesMed'].map({'No':0, 'Yes':1})


        dataset['max_glu_serum'] = dataset['max_glu_serum'].map({'None':0, '>300':1, 'Norm':2, '>200':3})
        dataset['A1Cresult'] = dataset['A1Cresult'].map({'None':0, '>7':1, '>8':2, 'Norm':3})


        dataset = dataset.drop(['repaglinide','nateglinide','chlorpropamide','acetohexamide','tolbutamide','miglitol','troglitazone',
        'tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
        'metformin-rosiglitazone','metformin-pioglitazone'],axis=1)


        #############################

        prediction = model.predict(dataset)

        if prediction == 0:
            prediction_text_val=' not Re-admit'
        
        elif prediction == 1:
            prediction_text_val=' re-admit after 30 days'
        
        elif prediction == 2:
            prediction_text_val=' re-admit before 30 days'

        #return render_template('Classification.html', prediction_text=prediction_text_val.format(prediction))

    except:
            return render_template('Classification.html', prediction_text='Invalid Encounter ID')
    #output = round(prediction[0] ) 
    return render_template('Classification.html', prediction_text='Patient might {}'.format(prediction_text_val),
                                                  encounterID='Encounter ID: {}'.format(encounterID),
                                                  patient_nbr='Patient No: {}'.format(patient_nbr),
                                                  payer_code='Payer code: {}'.format(payer_code))
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