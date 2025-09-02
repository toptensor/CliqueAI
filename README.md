# CliqueAI

CliqueAI - AI-Powered Maximum Clique Solver Network

## Maximum Clique Mechanism
Four-stage autonomous mechanism for distributed problem solving

### Problem Selection
Advanced AI algorithms curate complex graph problems from our distributed database. The system intelligently categorizes challenges by difficulty, graph structure, and computational requirements.

### Miner Selection
Smart allocation engine matches problems to miners based on experience levels and historical performance. Stake-weighted distribution ensures optimal resource utilization across the network.

### Scoring
Dual-metric evaluation system assesses both solution optimality and algorithmic diversity. This approach rewards accuracy while encouraging innovative problem-solving methodologies.

### Weight Setting
Exponential moving average algorithms continuously adjust miner reputation scores. Historical performance data influences future problem allocation and reward distribution.

## Documentation
- [Mechanism](docs/mechanism.md) – Detailed explanation of our mechanism.
- [W&B Logging](docs/wandb_logging.md) – Data logging schema for monitoring, analysis, and debugging via Weights & Biases.

## Getting Started
### Miner
```
./start_miner.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-miner-ip> --axon.port <your-miner-port>
```

### Validator
```
./start_validator.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-validator-ip> --axon.port <your-validator-port>
```