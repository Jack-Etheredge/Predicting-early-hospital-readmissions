from flask import Flask, render_template, flash, request, url_for, jsonify, session
from wtforms import Form, TextField, validators, SubmitField, SelectField
# # until we hook up the data science model prediction
import random

import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn import metrics

# from config import SECRET_KEY, DEBUG

#----------------- App config -----------------#
app = Flask(__name__)
# app.config.from_object(__name__)
# app.config['SECRET_KEY'] = SECRET_KEY
app.secret_key = "super secret key"


#----------------- Model and Scoring -----------------#
# types = ['Grass', 'Fire', 'Water', 'Fighting']
# attacks = ['Pound', 'Karate Chop', 'Double Slap', 'Comet Punch']

# def PredictClass(
#     attack1=None,
#     attack2=None,
#     gender=None):
#     # Run my model with the input data
#     return random.choice(types)

# (feature, (min, max), step)

# might need to turn this back on and mod it:
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

with open('randomforest.pkl', 'rb') as f:
    PREDICTOR = pickle.load(f)
    classes = PREDICTOR.classes_

def score(feature_stats_input):
    # return ', '.join([feature for feature in feature_stats_input])
    # data = request.json
    # x = np.matrix(data['example'])
    predictions = PREDICTOR.predict_proba(
        np.matrix(feature_stats_input))

    # return str(predictions)

    feats = {}
    for feature, importance in zip(classes, predictions[0]):
        feats[feature] = importance

    score = max(feats, key=lambda key: feats[key])

    # result = {"pokemon type": score}
    return score

#----------------- Forms -----------------#
# class ChoiceForms(Form):
#     frontend_selection = SelectField(
#         'Choose one:', choices=[('a', 'A'), ('b', 'B')],
#         validators=[validators.required()])

class PokeForm(Form):
    # this code was generated from (and then modified):
    # [print(ffs[0] + ' = SelectField(\'' + ' '.join([(s[0].upper() + s[1:]) for s in ffs[0].split('_')]) + '\', choices=[(str(x), str(x)) for x in range(' + str(ffs[1][0]) + ',' + str(ffs[1][1]) + ')], validators=[validators.required()])') for ffs in feature_form_settings]
    # generator doesn't handle the two that were decimals

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

        # against_bug = SelectField('Age', choices=[])
        # against_dark = SelectField('Against Dark', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_dragon = SelectField('Against Dragon', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_electric = SelectField('Against Electric', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_fairy = SelectField('Against Fairy', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_fight = SelectField('Against Fight', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_fire = SelectField('Against Fire', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_flying = SelectField('Against Flying', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_ghost = SelectField('Against Ghost', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_grass = SelectField('Against Grass', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_ground = SelectField('Against Ground', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_ice = SelectField('Against Ice', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_normal = SelectField('Against Normal', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_poison = SelectField('Against Poison', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_psychic = SelectField('Against Psychic', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_rock = SelectField('Against Rock', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_steel = SelectField('Against Steel', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # against_water = SelectField('Against Water', choices=[(str(x/10), str(x/10)) for x in range(0,21)], validators=[validators.required()])
        # pokedex_number = SelectField('Pokedex Number', choices=[(str(x), str(x)) for x in range(1,785)], validators=[validators.required()])
        # hp = SelectField('Hp', choices=[(str(x), str(x)) for x in range(20,255)], validators=[validators.required()])
        # against_rock = SelectField('Against Rock', choices=[(str(x), str(x)) for x in range(1,5)], validators=[validators.required()])
        # sp_defense = SelectField('Sp Defense', choices=[(str(x*5), str(x*5)) for x in range(4,46)], validators=[validators.required()])
        # defense = SelectField('Defense', choices=[(str(x*5), str(x*5)) for x in range(1,46)], validators=[validators.required()])
        # generation = SelectField('Generation', choices=[(str(x), str(x)) for x in range(1,7)], validators=[validators.required()])
        # percentage_male = SelectField('Percentage Male', choices=[(str(x), str(x)) for x in range(0,100)], validators=[validators.required()])
        # sp_attack = SelectField('Sp Attack', choices=[(str(x*5), str(x*5)) for x in range(2,35)], validators=[validators.required()])
        # base_total = SelectField('Base Total', choices=[(str(x*20), str(x*20)) for x in range(9,35)], validators=[validators.required()])
        # attack = SelectField('Attack', choices=[(str(x*5), str(x*5)) for x in range(1,37)], validators=[validators.required()])
        # speed = SelectField('Speed', choices=[(str(x*5), str(x*5)) for x in range(1,32)], validators=[validators.required()])
        #
        # # Because these are decimal based, we use ten times the desired value and handle the float conversion in the list comprehension
        # weight_choices = [(str(x), str(x)) for x in range(1,920)]
        # weight_choices.insert(0, ('0.1', '0.1'))
        # weight_kg = SelectField('Weight Kg', choices=weight_choices, validators=[validators.required()])
        # height_m = SelectField('Height M', choices=[(str(float(x)/10), str(float(x)/10)) for x in range(1,145)], validators=[validators.required()])
        #
        # # Because these numbers are too big we multiply them out from smaller base numbers
        # experience_growth = SelectField('Experience Growth', choices=[(str(x*10000), str(x*10000)) for x in range(60,164)], validators=[validators.required()])
        # base_egg_steps = SelectField('Base Egg Steps', choices=[(str(x*64), str(x*64)) for x in range(20,160)], validators=[validators.required()])

#----------------- App routes -----------------#
@app.route("/", methods=['GET', 'POST'])
def predict():
    form = PokeForm(request.form)
    print(form.errors)

    if request.method == 'POST':
        # created with generator:
        # [print(ffs[0] + ' = request.form[\'' + ffs[0] + '\']') for ffs in feature_form_settings]
        # against_bug = request.form['against_bug']
        # against_dark = request.form['against_dark']
        # against_dragon = request.form['against_dragon']
        # against_electric = request.form['against_electric']
        # against_fairy = request.form['against_fairy']
        # against_fight = request.form['against_fight']
        # against_fire = request.form['against_fire']
        # against_flying = request.form['against_flying']
        # against_ghost = request.form['against_ghost']
        # against_grass = request.form['against_grass']
        # against_ground = request.form['against_ground']
        # against_ice = request.form['against_ice']
        # against_normal = request.form['against_normal']
        # against_poison = request.form['against_poison']
        # against_psychic = request.form['against_psychic']
        # against_rock = request.form['against_rock']
        # against_steel = request.form['against_steel']
        # against_water = request.form['against_water']
        # pokedex_number = request.form['pokedex_number']
        # hp = request.form['hp']
        # base_egg_steps = request.form['base_egg_steps']
        # against_rock = request.form['against_rock']
        # sp_defense = request.form['sp_defense']
        # defense = request.form['defense']
        # generation = request.form['generation']
        # experience_growth = request.form['experience_growth']
        # percentage_male = request.form['percentage_male']
        # weight_kg = request.form['weight_kg']
        # height_m = request.form['height_m']
        # sp_attack = request.form['sp_attack']
        # base_total = request.form['base_total']
        # attack = request.form['attack']
        # speed = request.form['speed']

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

        # print(attack_selection)
        if form.validate():
            # Return the pokemon class type
            flash('Your patient is likely to be readmitted: ' + score(
                # [eval(feature[0]) for feature in feature_form_settings]
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
