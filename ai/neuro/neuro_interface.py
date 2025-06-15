# ai/neuro/neuro_interface.py

import datetime
import logging

class NeuroPersonalizationEngine:
    def __init__(self, advanced_threshold=0.7, fatigue_threshold=0.6, stress_threshold=0.6, logger=None):
        self.advanced_threshold = advanced_threshold
        self.fatigue_threshold = fatigue_threshold
        self.stress_threshold = stress_threshold
        self.logger = logger or logging.getLogger("NeuroPersonalizationEngine")
        self.logger.setLevel(logging.INFO)
        self.user_profiles = {}

    def analyze_brainwaves(self, user_brainwave_data):
        """
        Analyze brainwave data and return an experience mode and explanation.
        Expected keys: focus, relaxation, stress, fatigue (all 0..1)
        """
        focus = user_brainwave_data.get('focus', 0.0)
        relaxation = user_brainwave_data.get('relaxation', 0.0)
        stress = user_brainwave_data.get('stress', 0.0)
        fatigue = user_brainwave_data.get('fatigue', 0.0)

        if fatigue > self.fatigue_threshold:
            mode = "Activate rest mode"
            explanation = "Fatigue detected. Suggest rest or simplified interface."
        elif stress > self.stress_threshold:
            mode = "Enable stress-reducing guidance"
            explanation = "High stress detected. Activate calming features and step-by-step guidance."
        elif focus > self.advanced_threshold and relaxation > 0.5:
            mode = "Enable advanced trading interface"
            explanation = "User is highly focused and relaxed; unlock advanced tools."
        elif focus > 0.5:
            mode = "Enable standard interface"
            explanation = "User focus is moderate; standard features enabled."
        else:
            mode = "Enable guided mode"
            explanation = "Low focus; guided experience and hints enabled."

        self.logger.info(f"[{datetime.datetime.utcnow()}] Mode: {mode} | Details: {explanation}")
        return mode, explanation

    def personalize_experience(self, user_id, user_brainwave_data):
        """
        Personalizes the experience for a given user_id and stores profile.
        """
        mode, explanation = self.analyze_brainwaves(user_brainwave_data)
        self.user_profiles[user_id] = {
            "last_mode": mode,
            "last_update": datetime.datetime.utcnow().isoformat(),
            "data": user_brainwave_data
        }
        return mode, explanation

    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, None)

# Example usage:
# neuro = NeuroPersonalizationEngine()
# user_brainwave_data = {"focus": 0.8, "relaxation": 0.6, "stress": 0.2, "fatigue": 0.1}
# mode, explanation = neuro.personalize_experience("user123", user_brainwave_data)
# print(mode)
# print(explanation)
# print(neuro.get_user_profile("user123"))
