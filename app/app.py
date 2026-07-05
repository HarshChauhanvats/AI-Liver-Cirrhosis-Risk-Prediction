from flask import Flask, render_template, request
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Absolute model path
model_path = os.path.join(
    os.path.dirname(__file__),
    '../models/liver_rf_model.pkl'
)

model = joblib.load(model_path)
# Database setup

conn = sqlite3.connect('hospital.db')

cursor = conn.cursor()

cursor.execute('''

CREATE TABLE IF NOT EXISTS patients (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    age REAL,
    bilirubin REAL,
    cholesterol REAL,
    albumin REAL,
    copper REAL,
    alk_phos REAL,
    sgot REAL,
    tryglicerides REAL,
    platelets REAL,
    prothrombin REAL,

    sex_m INTEGER,
    ascites_y INTEGER,
    hepatomegaly_y INTEGER,
    spiders_y INTEGER,
    edema_s INTEGER,
    edema_y INTEGER,

    prediction TEXT,
    confidence REAL,

    created_at TEXT

)

''')

conn.commit()

conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['Age'])
    bilirubin = float(request.form['Bilirubin'])
    cholesterol = float(request.form['Cholesterol'])
    albumin = float(request.form['Albumin'])
    copper = float(request.form['Copper'])
    alk_phos = float(request.form['Alk_Phos'])
    sgot = float(request.form['SGOT'])
    tryglicerides = float(request.form['Tryglicerides'])
    platelets = float(request.form['Platelets'])
    prothrombin = float(request.form['Prothrombin'])

    sex_m = int(request.form['Sex_M'])
    ascites_y = int(request.form.get('Ascites_Y',0))
    hepatomegaly_y = int(request.form.get('Hepatomegaly_Y',0))
    spiders_y = int(request.form.get('Spiders_Y',0))
    edema_s = int(request.form.get('Edema_S',0))
    edema_y = int(request.form.get('Edema_Y',0))

    features = [
        age,
        bilirubin,
        cholesterol,
        albumin,
        copper,
        alk_phos,
        sgot,
        tryglicerides,
        platelets,
        prothrombin,
        sex_m,
        ascites_y,
        hepatomegaly_y,
        spiders_y,
        edema_s,
        edema_y
    ]
    
    # ------ ye hummne es warning UserWarning: X does not have valid feature names... ko hatane ke liye use kiya --------

    import pandas as pd
    
    # Give the list the exact names the model was trained on
    feature_names = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Sex_M', 'Ascites_Y', 'Hepatomegaly_Y', 'Spiders_Y', 'Edema_S', 'Edema_Y']
    
    # Convert to dataframe
    df_features = pd.DataFrame([features], columns=feature_names)
    
    probability = model.predict_proba(df_features)[0][1]
    prediction = model.predict(df_features)

    # ------------------------------------

    # ================= SHAP AI EXPLANATION =================
    import shap
    
    # 1. Ask SHAP to analyze your Random Forest model
    explainer = shap.TreeExplainer(model)
    
    # 2. Get the specific impacts for THIS patient's dataframe
    shap_values = explainer.shap_values(df_features)
    
    # 3. For Random Forest, shap_values[1] holds the "High Risk" weights
    # 3. Safely extract the impacts based on your SHAP version/model
    if isinstance(shap_values, list):
        # Older SHAP versions: Returns a list of arrays
        patient_impacts = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        # Newer SHAP versions: Returns a 3D array (samples, features, classes)
        patient_impacts = shap_values[0, :, 1]
    else:
        # Certain models (like XGBoost): Returns a 2D array of just the positive class
        patient_impacts = shap_values[0]
    
    # 4. Pair the feature names with their impact scores
    impact_dict = dict(zip(df_features.columns, patient_impacts))
    
    # 5. Sort them from highest impact to lowest
    sorted_impacts = sorted(impact_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 6. Grab the names of the top 2 contributing factors
    top_factor_1 = sorted_impacts[0][0].replace('_', ' ')
    top_factor_2 = sorted_impacts[1][0].replace('_', ' ')
    
    # 7. Create the Viva-Killer sentence!
    ai_explanation = f"The top contributing factors driving this prediction were {top_factor_1} and {top_factor_2}."
    # ========================================================

    if prediction[0] == 1:
        result = "High Risk of Liver Cirrhosis"
        risk_color = "#dc2626"
        recommendation = """
        Immediate hepatologist consultation recommended.
        Monitor bilirubin, albumin and liver enzymes regularly.
        Ultrasound and liver function tests advised.
        """
    else:
        result = "Low Risk of Liver Cirrhosis"
        risk_color = "#16a34a"

        recommendation = """
        Maintain healthy lifestyle and periodic monitoring.
        Continue regular liver health screening.
        """
    confidence = round(probability * 100, 2)

    # Save patient data into database

    conn = sqlite3.connect('hospital.db')

    cursor = conn.cursor()

    cursor.execute('''

    INSERT INTO patients (

        age,
        bilirubin,
        cholesterol,
        albumin,
        copper,
        alk_phos,
        sgot,
        tryglicerides,
        platelets,
        prothrombin,

        sex_m,
        ascites_y,
        hepatomegaly_y,
        spiders_y,
        edema_s,
        edema_y,

        prediction,
        confidence,
        created_at

    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    ''', (

        age,
        bilirubin,
        cholesterol,
        albumin,
        copper,
        alk_phos,
        sgot,
        tryglicerides,
        platelets,
        prothrombin,

        sex_m,
        ascites_y,
        hepatomegaly_y,
        spiders_y,
        edema_s,
        edema_y,

        result,
        confidence,

        datetime.now().strftime('%d-%m-%Y %H:%M')

    ))

    conn.commit()

    conn.close()

    import os

    os.system("python export_data.py")

    return render_template(
        'result.html',
        prediction_text=result,
        confidence=confidence,
        risk_color=risk_color,
        recommendation=recommendation,

        ai_explanation=ai_explanation,

        age=age,
        bilirubin=bilirubin,
        cholesterol=cholesterol,
        albumin=albumin,
        copper=copper,
        alk_phos=alk_phos,
        sgot=sgot,
        tryglicerides=tryglicerides,
        platelets=platelets,
        prothrombin=prothrombin
    )

@app.route('/dashboard')
def dashboard():

    import pandas as pd
    

    conn = sqlite3.connect('hospital.db')

    df = pd.read_sql_query(
        "SELECT * FROM patients",
        conn
    )

    conn.close()

    total_cases = len(df)

    print(df[['id','prediction','confidence']])

    avg_confidence = round(
        df['confidence'].mean(),
        2
    )

    avg_bilirubin = round(
        df['bilirubin'].mean(),
        2
    )

    high_risk = len(
        df[df['prediction'] ==
        'High Risk of Liver Cirrhosis']
    )

    low_risk = len(
        df[df['prediction'] ==
        'Low Risk of Liver Cirrhosis']
    )

    # PIE CHART DATAFRAME

    ids = [int(i) for i in df['id'].tolist()]
    confidences = [float(c) for c in df['confidence'].tolist()]
    colors = []

    for pred in df['prediction']:

        if pred == 'High Risk of Liver Cirrhosis':
            colors.append('#ef4444')

        else:
            colors.append('#22c55e')

    return render_template(

        'dashboard.html',

        total_cases=total_cases,

        avg_confidence=avg_confidence,

        avg_bilirubin=avg_bilirubin,

        high_risk=high_risk,

        low_risk=low_risk,

        ids=ids,

        confidences=confidences,

        colors=colors
    )

# yahan pe bulk prediction ke liye add kiya

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():

    file = request.files['file']

    import pandas as pd

    df = pd.read_csv(file)

    predictions = []

    confidences = []

    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()

    for _, row in df.iterrows():

        features = [[
            row['Age'],
            row['Bilirubin'],
            row['Cholesterol'],
            row['Albumin'],
            row['Copper'],
            row['Alk_Phos'],
            row['SGOT'],
            row['Tryglicerides'],
            row['Platelets'],
            row['Prothrombin'],
            row['Sex_M'],
            row['Ascites_Y'],
            row['Hepatomegaly_Y'],
            row['Spiders_Y'],
            row['Edema_S'],
            row['Edema_Y']
        ]]

        probability = model.predict_proba(features)[0][1]

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "High Risk of Liver Cirrhosis"
        else:
            result = "Low Risk of Liver Cirrhosis"

        confidence = round(probability * 100, 2)

        predictions.append(result)
        confidences.append(confidence)

        cursor.execute('''

        INSERT INTO patients (

            age,
            bilirubin,
            cholesterol,
            albumin,
            copper,
            alk_phos,
            sgot,
            tryglicerides,
            platelets,
            prothrombin,

            sex_m,
            ascites_y,
            hepatomegaly_y,
            spiders_y,
            edema_s,
            edema_y,

            prediction,
            confidence,
            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ''', (

            row['Age'],
            row['Bilirubin'],
            row['Cholesterol'],
            row['Albumin'],
            row['Copper'],
            row['Alk_Phos'],
            row['SGOT'],
            row['Tryglicerides'],
            row['Platelets'],
            row['Prothrombin'],

            row['Sex_M'],
            row['Ascites_Y'],
            row['Hepatomegaly_Y'],
            row['Spiders_Y'],
            row['Edema_S'],
            row['Edema_Y'],

            result,
            confidence,

            datetime.now().strftime('%d-%m-%Y %H:%M')

        ))

    conn.commit()
    conn.close()

    df['Prediction'] = predictions
    df['Confidence'] = confidences

    df.to_csv('patients_export.csv', index=False)

    os.system("python export_data.py")

    return df.to_html(classes='table table-bordered')


# ================= NEW SEARCH ROUTE =================

@app.route('/patient', methods=['GET'])
def patient_profile():
    # 1. Get the ID typed into the search bar
    patient_id = request.args.get('id')
    
    # 2. Connect to the database and search for that ID
    import sqlite3
    conn = sqlite3.connect('hospital.db')
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name (like a dictionary)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()
    conn.close()

    # 3. If the patient doesn't exist, show an error
    if not patient:
        return f"""
        <div style="text-align:center; margin-top:50px; font-family:Arial;">
            <h2 style="color:#ef4444;">Patient #{patient_id} Not Found</h2>
            <p>No records exist for this ID.</p>
            <a href="/dashboard" style="padding:10px 20px; background:#2563eb; color:white; text-decoration:none; border-radius:5px;">Go Back</a>
        </div>
        """

    # 4. If the patient exists, send their data to the profile page
    return render_template('patient_profile.html', patient=patient)
# =============================================================


if __name__ == '__main__':
    app.run(debug=True)