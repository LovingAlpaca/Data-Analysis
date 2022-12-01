import joblib as joblib
import sklearn
class CsPCa:
    def __init__(self):
        self.scaler = None
        self.predictor = None
        self.scaler_SB = None
        self.scaler_TB = None
        self.scaler_MRI = None
        self.predictor_SB = None
        self.predictor_TB = None
        self.predictor_MRI = None
    def load_models(self):
        print("load_models")
        model_dir="./"
        self.scaler = joblib.load(f"{model_dir}standardScaler_SBTB.joblib")
        self.predictor = joblib.load(f"{model_dir}Random_Models_SBTB.joblib")
        self.scaler_SB = joblib.load(f"{model_dir}standardScaler_SB.joblib")
        self.scaler_TB = joblib.load(f"{model_dir}standardScaler_TB.joblib")
        self.scaler_MRI = joblib.load(f"{model_dir}standardScaler_MRI.joblib")
        self.predictor_SB = joblib.load(f"{model_dir}Random_Models_SB.joblib")
        self.predictor_TB = joblib.load(f"{model_dir}Random_Models_TB.joblib")
        self.predictor_MRI = joblib.load(f"{model_dir}Random_Models_MRI.joblib")
csPCa = CsPCa()
csPCa.load_models()