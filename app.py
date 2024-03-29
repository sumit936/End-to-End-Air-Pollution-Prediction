from flask import Flask, render_template,request,redirect
from flask_sqlalchemy import SQLAlchemy
from src.Pipeline.predict_pipeline import CustomData,PredictPipeline
from numpy import round

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///prediction.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.app_context().push()

class Prediction(db.Model):
    SNo = db.Column(db.Integer,primary_key = True)
    Temperature = db.Column(db.Float,nullable = False)
    Humidity = db.Column(db.Float,nullable = False)
    Wind_Speed = db.Column(db.Float,nullable = False)
    Visibility = db.Column(db.Float,nullable = False)
    Pressure = db.Column(db.Float,nullable = False)
    so2 = db.Column(db.Float,nullable = False)
    no2 = db.Column(db.Float,nullable = False)
    Rainfall = db.Column(db.Float,nullable = False)
    PM10 = db.Column(db.Integer,nullable = False)
    PM25 = db.Column(db.Float,nullable = False)
    prediction = db.Column(db.Float,nullable = False)
    Quality = db.Column(db.String,nullable = False)

    def __repr__(self) -> str:
        return f"{self.SNo}-{self.prediction}"


@app.route('/', methods = ['GET', 'POST'])
def hello_world():

    if request.method == 'POST':
        Temp = request.form['Temperature']
        Hum = request.form['Humidity']
        WS = request.form['Wind speed']
        Vis = request.form['Visibility']
        Pres = request.form['Pressure']
        so2 = request.form['so2']
        no2 = request.form['no2']
        Rain = request.form['Rainfall']
        PM10 = request.form['PM10']
        PM25 = request.form['PM25']
        data = CustomData(float(Temp),
                          float(Hum),
                          float(WS),
                          float(Vis),
                          float(Pres),
                          float(so2),
                          float(no2),
                          float(Rain),
                          int(PM10),
                          float(PM25)
                         )
        
        df = data.get_data_as_data_frame()
        pred = PredictPipeline()
        p, quality = pred.predict(df)
        preds = round(p,2)
        PredObj = Prediction(
                            Temperature = Temp,
                            Humidity = Hum,
                            Wind_Speed = WS,
                            Visibility = Vis,
                            Pressure = Pres,
                            so2 = so2,
                            no2 = no2,
                            Rainfall = Rain,
                            PM10 = PM10,
                            PM25 = PM25,
                            prediction = preds,
                            Quality = quality
                            )
        db.session.add(PredObj)
        db.session.commit()
    allFeat = Prediction.query.all()
    return render_template('index.html', allFeat = allFeat)

@app.route('/update/<int:SNo>', methods = ['GET','POST'])
def update(SNo):
    if request.method == 'POST':
        Temp = request.form['Temperature']
        Hum = request.form['Humidity']
        WS = request.form['Wind speed']
        Vis = request.form['Visibility']
        Pres = request.form['Pressure']
        so2 = request.form['so2']
        no2 = request.form['no2']
        Rain = request.form['Rainfall']
        PM10 = request.form['PM10']
        PM25 = request.form['PM25']
        data = CustomData(float(Temp),
                          float(Hum),
                          float(WS),
                          float(Vis),
                          float(Pres),
                          float(so2),
                          float(no2),
                          float(Rain),
                          int(PM10),
                          float(PM25)
                         )
        
        df = data.get_data_as_data_frame()
        pred = PredictPipeline()
        p, quality = pred.predict(df)
        preds = round(p,2)
        predobj = Prediction(
                            Temperature = Temp,
                            Humidity = Hum,
                            Wind_Speed = WS,
                            Visibility = Vis,
                            Pressure = Pres,
                            so2 = so2,
                            no2 = no2,
                            Rainfall = Rain,
                            PM10 = PM10,
                            PM25 = PM25,
                            prediction = preds,
                            Quality = quality
                            )
        
        db.session.add(predobj)
        db.session.commit()
        return redirect("/")
    Feat = Prediction.query.filter_by(SNo = SNo).first()
    return render_template('update.html', Feat = Feat)
@app.route('/delete/<int:SNo>')
def delete(SNo):
    delobj = Prediction.query.filter_by(SNo = SNo).first()
    db.session.delete(delobj)
    db.session.commit()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug  = True, port=7000)