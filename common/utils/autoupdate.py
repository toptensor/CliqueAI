import subprocess
import traceback
import time

import bittensor as bt


def run_cmd(*args) -> str:
    """Run a command synchronously and return stdout as string."""
    try:
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command {' '.join(args)} failed: {e.stderr.decode().strip()}")


def get_local_hash() -> str:
    return run_cmd("git", "rev-parse", "HEAD")


def get_remote_hash(branch="origin/main") -> str:
    run_cmd("git", "fetch")
    return run_cmd("git", "rev-parse", branch)


def update_repo():
    run_cmd("git", "pull")


def update_repo_if_needed() -> bool:
    """Returns True if repo was updated and restart is needed, else False."""
    try:
        local_hash = get_local_hash()
        remote_hash = get_remote_hash(f"origin/main")

        if local_hash != remote_hash:
            bt.logging.info(f"Update available: {local_hash} -> {remote_hash}")
            bt.logging.info("Updating repository...")
            update_repo()
            bt.logging.info("Repository updated. Please restart the process.")
            return True
        else:
            bt.logging.info("No updates available.")
            return False

    except Exception as e:
        bt.logging.error(f"Error checking for updates: {e}")
        bt.logging.error(traceback.format_exc())
        time.sleep(60) # sleep 1 min
        return True
