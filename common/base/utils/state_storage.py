import json
import os
import sqlite3
from typing import Tuple

import numpy as np


def _ensure_db_schema(db_path: str):
    """
    Ensure that the SQLite database schema exists.
    If the database does not exist, it will be created with the necessary table.

    Args:
        db_path (str): The path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS validator_state (
            step INTEGER PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            scores BLOB,
            hotkeys BLOB,
            ema_step_count BLOB
        )
    """
    )
    conn.commit()
    conn.close()


def save_validator_state(
    path: str,
    step: int,
    ema_scores: np.ndarray,
    hotkeys: list[str],
    ema_step_count: np.ndarray,
):
    """
    Save the validator state to a SQLite database.

    Args:
        path (str): The path to the directory where the database will be stored.
        step (int): The current step of the validator.
        ema_scores (np.ndarray): The scores of the miners as a NumPy array.
        hotkeys (list[str]): The list of hotkeys associated with the miners.
        ema_step_count (np.ndarray): The EMA step count of the miners.
    """
    db_path = os.path.join(path, "validator_state.db")
    _ensure_db_schema(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ema_scores_json = json.dumps(ema_scores.tolist())
    hotkeys_json = json.dumps(hotkeys)
    ema_step_count_json = json.dumps(ema_step_count.tolist())

    cursor.execute(
        """
        INSERT OR REPLACE INTO validator_state (step, scores, hotkeys, ema_step_count)
        VALUES (?, ?, ?, ?)
    """,
        (step, ema_scores_json, hotkeys_json, ema_step_count_json),
    )

    conn.commit()
    conn.close()


def load_latest_validator_state(path: str) -> Tuple[int, np.ndarray, list[str]]:
    """Load the latest validator state from the database.

    Args:
        path (str): The path to the directory containing the database.

    Raises:
        FileNotFoundError: If the database file does not exist.
        ValueError: If the database is empty or the data is corrupted.

    Returns:
        Tuple[int, np.ndarray, list[str], np.ndarray]: A tuple containing the step, ema_scores, hotkeys, and ema_step_count.
    """
    db_path = os.path.join(path, "validator_state.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT step, scores, hotkeys, ema_step_count FROM validator_state ORDER BY timestamp DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        step, ema_scores_bytes, hotkeys_bytes, ema_step_count_bytes = row
        ema_scores = np.array(json.loads(ema_scores_bytes), dtype=np.float32)
        hotkeys = json.loads(hotkeys_bytes)
        ema_step_count = np.array(json.loads(ema_step_count_bytes), dtype=np.int32)
        return step, ema_scores, hotkeys, ema_step_count
    else:
        raise ValueError("No validator state found in the database.")


def get_all_validator_state(path: str) -> list[Tuple[int, list[float], list[str]]]:
    """
    Get all validator states from the database.

    Args:
        path (str): The path to the directory containing the database.

    Returns:
        list[Tuple[int, list[float], list[str]]]: A list of tuples containing step, scores, and hotkeys.
    """
    db_path = os.path.join(path, "validator_state.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT step, scores, hotkeys FROM validator_state")
    rows = cursor.fetchall()
    conn.close()

    return [(row[0], json.loads(row[1]), json.loads(row[2])) for row in rows]
