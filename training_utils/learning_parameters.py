# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

# defenseur :
BATCH_SIZE = 2048

# Fonction Q = Score(t) + Q(t-1)
GAMMA_START = 0.95
GAMMA_END = 0.95
GAMMA_OFFSET = 0
GAMMA_TEMPS = 1 # Au bout de n+GAMMA_OFFSET événements, on serra a GAMMA_START + (GAMMA_END - GAMMA_START)*(n/GAMMA_TEMPS)

# Fréquence des actions aléatoires
#  defense
EPS_START_DFS = 0.90
EPS_END_DFS = 0.05
EPS_OFFSET_DFS = 10000
EPS_DECAY_DFS = 40000

#  actor
EPS_START_ATQ = 0.10
EPS_END_ATQ = 0.05
EPS_OFFSET_ATQ = 0
EPS_DECAY_ATQ = 100000

# Taux pour la soft actualisation (stabilise le modèle)
TAU = 0.002

# Learning rates
LR_ACTOR = 5e-5
LR_CRITIC = 5e-5

