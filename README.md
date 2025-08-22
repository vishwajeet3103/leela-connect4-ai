# Connect Four AI (AlphaZero Style)

This project is an implementation of an AlphaZero-style AI for Connect Four.  
It uses Deep Reinforcement Learning with Monte Carlo Tree Search (MCTS) to learn how to play through self-play.  
Built with PyTorch, you can train the model and even play against it.

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python connect4_ai.py
```

## Play Against AI
```bash
python connect4_ai.py --play --model_in c4_az.pt
```

## License
MIT License
