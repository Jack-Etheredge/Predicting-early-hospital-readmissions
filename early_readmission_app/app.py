"""Flask app that predicts whether a patient will be readmitted early to the hospital (<30 days)"""

from pathlib import Path

from flask import Flask, render_template, flash, request, session
from wtforms import Form, validators, SelectField
import random
import numpy as np
import pandas as pd
import pickle

APP_DIR = Path(__file__).parent

app = Flask(__name__)
app.secret_key = "super secret key"

feature_form_settings = [
    ('num_lab_procedures', (1, 133), 1),
    ('number_inpatient', (0,22), 1),
    ('num_medications', (1,133), 1),
    ('time_in_hospital', (1, 15), 1),
    ('number_diagnoses', (1, 17), 1),
    ('num_procedures', (0, 7), 1),
    ('number_emergency', (0, 77), 1),
    ('number_outpatient', (0, 43), 1),
    ('Discharged_to_home', (0,2), 1),
    ('Dicshared_to_rehab_fac', (0,2), 1),
    ('gender_Male', (0,2), 1),
    ('payer_code_MC', (0,2), 1),
    ('diag_1_Circulatory_System', (0,2), 1),
    ('diag_2_Circulatory_System', (0,2), 1),
    ('diag_3_Circulatory_System', (0,2), 1),
    ('age_70_80', (0,2), 1),
    ('age_60_70', (0,2), 1),
    ('race_Caucasian', (0,2), 1),
    ('change_med_no', (0,2), 1),
    ('age_80_90', (0,2), 1),
    ('medical_specialty_InternalMedicine', (0,2), 1),
    ('admission_type_id_Emergency', (0,2), 1),
    ('diag_3_External_Causes', (0,2), 1),
    ('insulin_no', (0,2), 1),
    ('insulin_Steady', (0,2), 1)
]

model_path = APP_DIR / 'randomforest.pkl'
with open(model_path, 'rb') as f:
    PREDICTOR = pickle.load(f)
    classes = PREDICTOR.classes_


def score(feature_stats_input):
    predictions = PREDICTOR.predict_proba(
        np.matrix(feature_stats_input))

    feats = {}
    for feature, importance in zip(classes, predictions[0]):
        feats[feature] = importance

    score = max(feats, key=lambda key: feats[key])

    return score


class Form(Form):
        num_lab_procedures = SelectField('number of lab procedures',choices=[(str(x), str(x)) for x in range(1,133)], validators=[validators.required()])
        number_inpatient = SelectField('number of inpatient visits in the past year',choices=[(str(x), str(x)) for x in range(0,22)], validators=[validators.required()])
        num_medications = SelectField('number of medications',choices=[(str(x), str(x)) for x in range(1,133)], validators=[validators.required()])
        time_in_hospital = SelectField('time in hospital (days)',choices=[(str(x), str(x)) for x in range(1,15)], validators=[validators.required()])
        number_diagnoses = SelectField('number of diagnoses',choices=[(str(x), str(x)) for x in range(1,17)], validators=[validators.required()])
        num_procedures = SelectField('number of procedures',choices=[(str(x), str(x)) for x in range(0,7)], validators=[validators.required()])
        number_emergency = SelectField('number of emergency visits in the past year',choices=[(str(x), str(x)) for x in range(0,77)], validators=[validators.required()])
        number_outpatient = SelectField('number of outpatient visits in the past year',choices=[(str(x), str(x)) for x in range(0,43)], validators=[validators.required()])
        Discharged_to_home = SelectField('Discharged to home', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        Dicshared_to_rehab_fac = SelectField('Discharged/transferred to another rehab fac including rehab units of a hospital', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        gender_Male = SelectField('gender = male', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        payer_code_MC = SelectField('payer code = MC (Medicaid)', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        diag_1_Circulatory_System = SelectField('Diagnosis 1 = Diseases of the Circulatory System', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        diag_2_Circulatory_System = SelectField('Diagnosis 2 = Diseases of the Circulatory System', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        diag_3_Circulatory_System = SelectField('Diagnosis 3 = Diseases of the Circulatory System', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        age_70_80 = SelectField('Age = [70-80)', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        age_60_70 = SelectField('Age = [60-70)', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        race_Caucasian = SelectField('race = Caucasian', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        change_med_no = SelectField('No change in medication', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        age_80_90 = SelectField('Age = [80-90)', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        medical_specialty_InternalMedicine = SelectField("Physician's medical specialty = Internal Medicine", choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        admission_type_id_Emergency = SelectField('Admission type = Emergency', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        diag_3_External_Causes = SelectField('Diagnosis 3 = Supplementary Classification of External Causes of Injury And Poisoning', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        insulin_no = SelectField('Receiving insulin = no', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])
        insulin_Steady = SelectField('Insulin = steady', choices=[(str(x), str(x)) for x in range(0,2)], validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def predict():
    form = Form(request.form)
    print(form.errors)

    if request.method == 'POST':
        num_lab_procedures = request.form['num_lab_procedures']
        number_inpatient = request.form['number_inpatient']
        num_medications = request.form['num_medications']
        time_in_hospital = request.form['time_in_hospital']
        number_diagnoses = request.form['number_diagnoses']
        num_procedures = request.form['num_procedures']
        number_emergency = request.form['number_emergency']
        number_outpatient = request.form['number_outpatient']
        Discharged_to_home = request.form['Discharged_to_home']
        Dicshared_to_rehab_fac = request.form['Dicshared_to_rehab_fac']
        gender_Male = request.form['gender_Male']
        payer_code_MC = request.form['payer_code_MC']
        diag_1_Circulatory_System = request.form['diag_1_Circulatory_System']
        diag_2_Circulatory_System = request.form['diag_2_Circulatory_System']
        diag_3_Circulatory_System = request.form['diag_3_Circulatory_System']
        age_70_80 = request.form['age_70_80']
        age_60_70 = request.form['age_60_70']
        race_Caucasian = request.form['race_Caucasian']
        change_med_no = request.form['change_med_no']
        age_80_90 = request.form['age_80_90']
        medical_specialty_InternalMedicine = request.form['medical_specialty_InternalMedicine']
        admission_type_id_Emergency = request.form['admission_type_id_Emergency']
        diag_3_External_Causes = request.form['diag_3_External_Causes']
        insulin_no = request.form['insulin_no']
        insulin_Steady = request.form['insulin_Steady']

        if form.validate():
            flash('Your patient is likely to be readmitted: ' + score(
                [num_lab_procedures,
                number_inpatient,
                num_medications,
                time_in_hospital,
                number_diagnoses,
                num_procedures,
                number_emergency,
                number_outpatient,
                Discharged_to_home,
                Dicshared_to_rehab_fac,
                gender_Male,
                payer_code_MC,
                diag_1_Circulatory_System,
                diag_2_Circulatory_System,
                diag_3_Circulatory_System,
                age_70_80,
                age_60_70,
                race_Caucasian,
                change_med_no,
                age_80_90,
                medical_specialty_InternalMedicine,
                admission_type_id_Emergency,
                diag_3_External_Causes,
                insulin_no, insulin_Steady]))
        else:
            flash('All the form fields are required. ')

    return render_template('index.html',
        form=form,
        feature_form_settings=feature_form_settings)

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002,debug=True)
