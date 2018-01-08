import pickle
import os
from keras.models import save_model
BASE_PROJECT_PATH = '.'


def save(model, mapping, model_name):
    model_yaml_path = 'models/baseline_nn/baseline_nn.yaml'
    mapping_model_path = 'models/baseline_nn/baseline_nn_mapping.p'

    model_yaml = model.to_yaml()
    with open(model_yaml_path, "w") as yaml_file:
        yaml_file.write(model_yaml)

    save_model(model, model_h5_path)

    pickle.dump(mapping, open(mapping_model_path, 'wb'))

    return
