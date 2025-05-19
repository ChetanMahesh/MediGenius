from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, send_file
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import pandas as pd
import imutils
import sklearn
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import google.generativeai as genai
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Loading Models disease
covid_model = load_model('models/covid.h5')
braintumor_model = load_model('models/braintumor.h5')
alzheimer_model = load_model('models/alzheimer_model.h5')
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
pneumonia_model = load_model('models/pneumonia_model.h5')
#pharmacy
medicines_dict = pickle.load(open('models/medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('models/similarity.pkl', 'rb'))
#doctor recommendation
training_dataset = pd.read_csv('Doctor/Training.csv')
doc_dataset = pd.read_csv('Doctor/doctors_dataset.csv', names=['Name', 'Description'])

symptoms = list(training_dataset.columns[:-1])

# Preprocess the training data
X = training_dataset.iloc[:, :-1].values
y = training_dataset.iloc[:, -1].values

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Prepare doctor data
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()
diseases = pd.DataFrame(dimensionality_reduction.index, columns=['prognosis'])

doctors = pd.DataFrame()
doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']
doctors['disease'] = diseases['prognosis']


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

genai.configure(api_key="AIzaSyCCbalsvgxOoxu-0zLTnlbRTQWkWVs77ME")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Create the model of Medical Chatbot
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)
def medical_chatbot(input_text):
    response = model.generate_content([
    "you are a healthcare chatbot, so reply accordingly",
    f"input: {input_text}",
    "output: ",
    ])

    return response.text


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

def recommend(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = []
    for i in medicines_list:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
    return recommended_medicines

########################### Routing Functions ########################################


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')



@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')



@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')

@app.route('/chatbot')
def chat():
    return send_file('templates/chat.html')


@app.route('/doctor')
def doctor():
    return render_template('doctor.html')


########################### Result Functions ########################################


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)



@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)
    

#phramacy    
@app.route('/drug', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_medicine_name = None
    if request.method == 'POST':
        #recommend(medicine)
        selected_medicine_name = request.form['medicine']
        recommendations = recommend(selected_medicine_name)
        print(medicines_dict)
    return render_template('index.html', medicines=medicines['Drug_Name'].values, recommendations=recommendations, selected_medicine_name=selected_medicine_name)


#chatbot
@app.route('/chat', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get("message", "")
    bot_response = medical_chatbot(user_message)
    return jsonify({"response": bot_response})

#doctor recommendation
# Endpoint to fetch symptoms list
@app.route('/get_symptom_list', methods=['GET'])
def get_symptom_list():
    return jsonify({"symptoms": symptoms})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'responses' not in data:
        return jsonify({"error": "Invalid request: No responses provided"}), 400

    user_responses = data['responses']
    if not isinstance(user_responses, list) or not all(isinstance(r, int) for r in user_responses):
        return jsonify({"error": "Invalid responses format"}), 400

    # Pad responses to match the number of symptoms (132)
    total_symptoms = len(symptoms)  # Use the global `symptoms` list
    if len(user_responses) > total_symptoms:
        return jsonify({"error": "Too many responses provided"}), 400

    # Pad with zeros if fewer responses are provided
    padded_responses = user_responses + [0] * (total_symptoms - len(user_responses))
    responses = np.array(padded_responses).reshape(1, -1)

    # Predict using the classifier
    prediction = classifier.predict(responses)
    disease = labelencoder.inverse_transform(prediction)[0]

    # Get doctor recommendation
    doctor_info = doctors[doctors['disease'] == disease].iloc[0]
    return jsonify({
        "disease": disease,
        "doctor_name": doctor_info['name'],
        "doctor_link": doctor_info['link']
    })




# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response



if __name__ == '__main__':
    app.run(debug=True)
