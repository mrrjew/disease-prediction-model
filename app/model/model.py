import pickle
from pathlib import Path
import re

__version__ = "1.6.1"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/model.pkl", "rb") as f:
    model = pickle.load(f)

def predictDisease(symptoms):
  symptoms = symptoms.split(",")

  input_data = [0] * len(data_dict["symptom_index"])
  for symptom in symptoms:
    index = data_dict["symptom_index"][symptom]
    input_data[index] = 1

  input_data = np.array(input_data).reshape(1,-1)

  rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
  nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
  svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

  import statistics
  final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
  predictions = {
      "rf_model_prediction": rf_prediction,
      "naive_bayes_prediction": nb_prediction,
      "svm_model_prediction": svm_prediction,
      "final_prediction":final_prediction
  }
  return predictions