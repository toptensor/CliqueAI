import bittensor as bt


def _timestamp_text(timestamp: int | float) -> str:
    if isinstance(timestamp, float) and timestamp.is_integer():
        return str(int(timestamp))
    return str(timestamp)


def _signature_message(timestamp: int | float, hotkey: str) -> str:
    return f"{_timestamp_text(timestamp)}:{hotkey}"


def verify_signature(signature: str, timestamp: int | float, hotkey: str) -> bool:
    message = _signature_message(timestamp, hotkey)
    keypair = bt.Keypair(ss58_address=hotkey)
    try:
        return bool(keypair.verify(signature=signature, message=message))
    except TypeError:
        try:
            return bool(keypair.verify(message, bytes.fromhex(signature)))
        except Exception:
            return False
    except Exception:
        return False


def create_signature(wallet: bt.Wallet, timestamp: int) -> str:
    hotkey = wallet.hotkey.ss58_address
    signature = wallet.hotkey.sign(_signature_message(timestamp, hotkey))
    return signature.hex() if isinstance(signature, bytes) else str(signature)
