import numpy as np


class MedicalECGFeatures():
    def __init__(self, ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates, fs):
        # Set parameters
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        # Feature dictionary
        self.medical_features = dict()

    def get_medical_features(self):
        return self.medical_features

    def extract_medical_features(self):
        self.medical_features.update(self.calculate_medical_features())

    def calculate_medical_features(self):
        medical_features = dict()
        # Calculate statistics
        medical_features['QRS_height'] = self.calculate_qrs_height()
        medical_features['RR_interval'] = self.calculate_rri_intervals()
        return medical_features

    def calculate_qrs_height(self, window_before=50, window_after=50, threshold_std=2):
        qrs_heights = []

        for rpeak in self.rpeaks:
            # Define the window around the R-peak
            window_start = max(0, rpeak - window_before)
            window_end = min(len(self.signal_filtered), rpeak + window_after)

            # Extract the ECG signal within the window
            qrs_window = self.signal_filtered[window_start:window_end]

            # Calculate the amplitude of the signal within the window
            qrs_height = np.max(qrs_window) - np.min(qrs_window)

            # Append the QRS height to the list
            qrs_heights.append(qrs_height)

        # Calculate mean and standard deviation of QRS heights
        mean_qrs_height = np.mean(qrs_heights)
        std_qrs_height = np.std(qrs_heights)

        # Filter out abnormal values (those deviating too much from the mean)
        qrs_heights_filtered = [qrs_height for qrs_height in qrs_heights if
                                np.abs(qrs_height - mean_qrs_height) <= threshold_std * std_qrs_height]

        # Take the average of the remaining QRS heights
        if len(qrs_heights_filtered) > 0:
            avg_qrs_height = np.mean(qrs_heights_filtered)
        else:
            avg_qrs_height = np.nan

        return avg_qrs_height

    def calculate_rri_intervals(self, threshold_std=2):
        # Assume rpeaks contains the indices of R-peaks in the ECG signal
        rr_intervals = []

        for i in range(1, len(self.rpeaks)):
            # Calculate R-R interval as the time difference between consecutive R-peaks
            rr_interval = self.rpeaks[i] - self.rpeaks[i - 1]

            # Append R-R interval to the list
            rr_intervals.append(rr_interval)
        mean_rr_interval = np.mean(rr_intervals)
        std_rr_interval = np.std(rr_intervals)

        # Filter out abnormal values (those deviating too much from the mean)
        rr_intervals_filtered = [rr for rr in rr_intervals if
                                 np.abs(rr - mean_rr_interval) <= threshold_std * std_rr_interval]

        # Take the average of the remaining R-R intervals
        if len(rr_intervals_filtered) > 0:
            avg_rr_interval = np.mean(rr_intervals_filtered)
        else:
            avg_rr_interval = np.nan
        return avg_rr_interval/self.fs

    # Add more methods for other medical ECG features as needed
