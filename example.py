# Import 3rd party libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
# Import local Libraries
sys.path.insert(0, os.path.dirname(os.getcwd()))
from features.feature_extractor import Features
from utils.plotting.waveforms import plot_waveforms
from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/api/example', methods=['GET','POST'])
def example_post_api():
    # 处理 POST 请求的逻辑
    # req_data = request.get_json()
    # Sampling frequency (Hz)
    fs = 300

    # Data paths
    waveform_path = 'data/waveforms'
    feature_path = 'data/features'

    # Read labels CSV
    labels = pd.read_csv('data/labels/labels.csv', names=['file_name', 'label'])

    # View DataFrame
    labels.head(10)

    plot_waveforms(labels=labels, waveform_path=waveform_path, fs=fs)
    # Instantiate
    ecg_features = Features(file_path=waveform_path, fs=fs, feature_groups=['full_waveform_features'])

    # Calculate ECG features
    ecg_features.extract_features(
        filter_bandwidth=[3, 45], n_signals=None, show=True,
        labels=labels, normalize=True, polarity_check=True,
        template_before=0.25, template_after=0.4
    )
    # Get features DataFrame
    features = ecg_features.get_features()
    # View DataFrame
    features.head(10)
    # Save features DataFrame to CSV
    # features.to_csv(os.path.join(feature_path, 'features.csv'), index=False)
    features_json = features.to_json(orient='records')

    # 返回 JSON 字符串作为 API 响应
    return jsonify(features_json)


#
if __name__ == '__main__':
    app.run(debug=True)
