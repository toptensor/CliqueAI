import bittensor as bt


def verify_signature(signature: str, timestamp: int, hotkey: str) -> bool:
    message = f"{timestamp}:{hotkey}"
    keypair = bt.Keypair(ss58_address=hotkey)
    return keypair.verify(signature=signature, message=message)


def create_signature(wallet: bt.Wallet, timestamp: int) -> str:
    message = f"{timestamp}:{wallet.hotkey}"
    return wallet.hotkey.sign(message)
