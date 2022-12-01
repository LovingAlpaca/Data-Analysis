from flask import Flask, render_template, request ,url_for
import pandas as pd
import Model
app = Flask('csPCa_predict')

def get_predict_result(df,Select):
    #datajson = """{"Age":{"107":58.0},"PSA":{"107":9.86},"Volume":{"107":50.0},"PSAD":{"107":0.1972},"PIRADS_First":{"107":3.0},"ISUP":{"107":0.0},"PIRADS_Biopsy":{"107":3.0},"Gleason_SB_TB":{"107":0},"Gleason_TB":{"107":0.0},"Gleason_SB":{"107":0.0},"Scanner":{"107":1.0},"IL_Localisation":{"107":3},"IL_Diameter":{"107":10.0},"csPCa":{"107":0}}"""
    #df = pd.read_json(datajson)
    num_cols_SBTB = ['Age', 'PSA', 'Volume', 'PSAD', 'PIRADS_First', 'PIRADS_Biopsy', 'Gleason_SB_TB', 'Gleason_TB',
                     'Gleason_SB', 'Scanner', 'IL_Localisation', 'IL_Diameter']
    num_cols_SB = ['Age','PSA','Volume','PSAD','PIRADS_First','Gleason_SB','IL_Localisation','IL_Diameter']
    num_cols_TB = ['Age','PSA','Volume','PSAD','PIRADS_First','PIRADS_Biopsy','Gleason_TB','Scanner','IL_Localisation','IL_Diameter']
    num_cols_MRI = ['Age','PSA','Volume','PSAD','PIRADS_First','IL_Localisation','IL_Diameter']
    if Select == 1:
        num_cols = num_cols_MRI
        load_num_features = Model.csPCa.scaler_MRI.transform(df[num_cols])
        result = Model.csPCa.predictor_MRI.predict(load_num_features)
    if Select == 2:
        num_cols = num_cols_SB
        load_num_features = Model.csPCa.scaler_SB.transform(df[num_cols])
        result = Model.csPCa.predictor_SB.predict(load_num_features)
    if Select == 3:
        num_cols = num_cols_TB
        load_num_features = Model.csPCa.scaler_TB.transform(df[num_cols])
        result = Model.csPCa.predictor_TB.predict(load_num_features)
    if Select == 4:
        num_cols = num_cols_SBTB
        load_num_features = Model.csPCa.scaler.transform(df[num_cols])
        result = Model.csPCa.predictor.predict(load_num_features)
    return result

@app.route("/",methods=["get","post"])
def Home():
    return render_template("index.html")

@app.route("/MRI",methods=["get","post"])
def predict_MRI():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        df = pd.DataFrame([data.values()],columns=data.keys())
        result = str(get_predict_result(df,1))
    return render_template("MRI.html",result=result)

@app.route("/SB",methods=["get","post"])
def predict_SB():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        df = pd.DataFrame([data.values()],columns=data.keys())
        result = str(get_predict_result(df,2))
    return render_template("SB.html",result=result)

@app.route("/TB",methods=["get","post"])
def predict_TB():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        df = pd.DataFrame([data.values()],columns=data.keys())
        result = str(get_predict_result(df,3))
    return render_template("TB.html",result=result)

@app.route("/SBTB",methods=["get","post"])
def predict_SBTB():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        df = pd.DataFrame([data.values()],columns=data.keys())
        result = str(get_predict_result(df,4))
    return render_template("SBTB.html",result=result)

app.run("0.0.0.0",80,debug=True)