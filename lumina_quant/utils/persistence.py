import json
import logging
import os


class StateManager:
    """Manages the persistence of trading state (Positions, Strategy Logic) to a JSON file.
    Ensures that if the bot crashes and restarts, it resumes with correct context.
    """

    def __init__(self, file_path="state.json"):
        self.file_path = file_path
        self.logger = logging.getLogger("lumina_quant.state_manager")

    def load_state(self):
        """Loads the state from the JSON file.
        Returns empty dict if file doesn't exist or error.
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                return {}
        return {}

    def save_state(self, state_dict):
        """Saves the state dict to the JSON file."""
        try:
            # Atomic write (write to temp then rename) to prevent corruption
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(state_dict, f, indent=4)
            os.replace(temp_path, self.file_path)
            self.logger.debug("State saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
