import environment.game.main as game
import environment.action_space as action_space
import environment.ressources.maps as maps

def reset(choose_map = -1):
    global jeu, carte_cp, score_adv
    # Charge une nouvelle partie aléatoire
    # choisir une carte aleatoirement :
    
    if choose_map < 0:
        carte_cp, score_adv = maps.get_random_map()
    else: 
        carte_cp = maps.echantillon_carte[choose_map]
        score_adv = 1

    jeu = game.Etat_de_jeu(carte_cp, nb_tour=100)
    observation_hero, observation_adversaire = jeu.get_observation(indice=0), jeu.get_observation(indice=2)
    
    return observation_hero, observation_adversaire, False

def transform_output_1_pod(v, a, b, s):
    pow = min(100, max(0, int(100*((v-0.5)*1.2 + 0.5))))
    if b>0.95:
        pow = 'BOOST'
    if s>0.95:
        pow = 'SHIELD'
    ang = int(18*(2*a-1))

    return ang, pow

def transform_output(action):
    a,b,c,d,e,f,g,h = action
    ang1, pow1 = transform_output_1_pod(a, b, c, d)
    ang2, pow2 = transform_output_1_pod(e, f, g, h)
    return ang1, pow1, ang2, pow2
    

def step(action_hero, action_adversaire, factor_reduction_reward_adverse=1):
    global jeu
    
    observation_hero, observation_adversaire, reward1, reward2, terminated = jeu.etape_de_jeu(transform_output(action_hero), transform_output(action_adversaire))

    return observation_hero, observation_adversaire, reward1, reward2, terminated, False

def get_cp():
    global carte_cp, score_adv
    n_cp = len(carte_cp)
    n_cp_hero = jeu.get_n_cp(joueur=0)
    n_cp_adversaire = jeu.get_n_cp(joueur=2)
    print(n_cp_hero, n_cp_adversaire)
    return n_cp_hero/score_adv, n_cp_adversaire/score_adv


def afficher():
    global jeu

    jeu.afficher()