import json
import logging
import os
import tempfile
import threading


class StateManager:
    """Manages the persistence of trading state (Positions, Strategy Logic) to a JSON file.
    Ensures that if the bot crashes and restarts, it resumes with correct context.
    """

    def __init__(self, file_path="data/state.json"):
        self.file_path = file_path
        self.logger = logging.getLogger("lumina_quant.state_manager")
        # Serialize concurrent save_state calls.  _save_state is invoked from the main
        # loop AND the user-stream thread's order-state callback; without this lock two
        # writers racing on a fixed temp path corrupt the recovery file.
        self._save_lock = threading.Lock()

    def load_state(self):
        """Loads the state from the JSON file.
        Returns empty dict if file doesn't exist or error.
        """
        candidates = [str(self.file_path)]
        if str(self.file_path).replace("\\", "/") == "data/state.json":
            candidates.append("state.json")

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                return {}
        return {}

    def save_state(self, state_dict):
        """Saves the state dict to the JSON file.

        Atomic + concurrency-safe: a per-call UNIQUE temp file (mkstemp) is written
        and os.replace'd into place under a dedicated lock so two threads cannot
        clobber a shared temp path and leave a half-written recovery file.
        """
        with self._save_lock:
            tmp_fd = -1
            tmp_path = ""
            try:
                parent = os.path.dirname(str(self.file_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                # Per-call unique temp name in the SAME directory as the target so the
                # final os.replace is an atomic same-filesystem rename.
                target_dir = parent or "."
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=target_dir,
                    prefix=os.path.basename(str(self.file_path)) + ".",
                    suffix=".tmp",
                )
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    tmp_fd = -1  # ownership transferred to the file object
                    json.dump(state_dict, f, indent=4)
                os.replace(tmp_path, self.file_path)
                tmp_path = ""
                self.logger.debug("State saved successfully.")
            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
            finally:
                if tmp_fd != -1:
                    try:
                        os.close(tmp_fd)
                    except Exception:
                        pass
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
