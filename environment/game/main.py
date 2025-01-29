import environment.game.params.game_parametre as parametre
from environment.game.game_main_functions import pods_start, deplacement, cp_valide, fin
from environment.game.affichage_graphique import affgame
from environment.game.calcul_angle import normalized, norme_angle, next_item, angle, norme
from math import pi, cos, sin, sqrt, exp


def norm_ang(a):
    return normalized(a)/18
def norm_dist(d):
    return d/5000
def norm_vit(v):
    return v/500


class Etat_de_jeu:
    def __init__(self, carte_cp, nb_tour, cp_avant_teleportation = 100, entrainement_attaque = 1):
        self.pods = pods_start(carte_cp)

        self.carte_cp = carte_cp
        self.nb_tour = nb_tour
        self.cp_avant_teleportation = cp_avant_teleportation
        self.entrainement_attaque = entrainement_attaque

        self.memoire = [[self.pods[1].copy(), self.pods[2].copy(), self.pods[3].copy(), self.pods[4].copy()]]

        self.premier_tour = True

        self.tick = 0

        self.barycentre = self.find_barycentre()

    def find_barycentre(self):
        sum_x, sum_y = 0, 0
        for x, y in self.carte_cp:
            sum_x+=x
            sum_y+=y
        n=len(self.carte_cp)
        return sum_x/n, sum_y/n
    
    def etape_de_jeu(self, action_j1, action_j2):

        ang1a, pow1a, ang1b, pow1b = action_j1
        ang2a, pow2a, ang2b, pow2b = action_j2

        self.tick+=1
        bit_fin, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2 = self.pods
        if bit_fin == 0:

            reponse_j1_a, reponse_j1_b = (pod_j1_a[2] + 50000 * cos((ang1a+pod_j1_a[6])*pi/180), pod_j1_a[3] + 50000 * sin((ang1a+pod_j1_a[6])*pi/180), pow1a), (pod_j1_b[2] + 50000 * cos((ang1b+pod_j1_b[6])*pi/180), pod_j1_b[3] + 50000 * sin((ang1b+pod_j1_b[6])*pi/180), pow1b)
            reponse_j2_a, reponse_j2_b = (pod_j2_a[2] + 50000 * cos((ang2a+pod_j2_a[6])*pi/180), pod_j2_a[3] + 50000 * sin((ang2a+pod_j2_a[6])*pi/180), pow2a), (pod_j2_b[2] + 50000 * cos((ang2b+pod_j2_b[6])*pi/180), pod_j2_b[3] + 50000 * sin((ang2b+pod_j2_b[6])*pi/180), pow2b)

            lpod = [(pod_j1_a, reponse_j1_a), (pod_j1_b, reponse_j1_b), (pod_j2_a, reponse_j2_a), (pod_j2_b, reponse_j2_b)]
            
    
            
            # la fonction deplacement() modifie en place la case 6 = orientation de chaque pod
            # la fonction deplacement() modifie en place les cases 4 et 5 = vitesse de chaque pod
            # la fonction deplacement() modifie en place les cases 2 et 3 = position de chaque pod

            boost_j1, boost_j2, rebond_fratricide, rebond_ennemi = deplacement(lpod, self.premier_tour, boost_j1, boost_j2, self.entrainement_attaque) 


            cp_valide(lpod, self.carte_cp, self.cp_avant_teleportation, self.entrainement_attaque) # modifie en place les cases 0 (nb de tour) 1 (prochain cp) et 7 (nb de tour sans passer de cp) de chaque pod

            # Change le bit de fin de manière 
            if parametre.defaite_j2_possible and (pod_j2_a[7]>parametre.nombre_de_tick_max_sans_cp and pod_j2_b[7]>parametre.nombre_de_tick_max_sans_cp) or (pod_j1_a[0] == self.nb_tour and pod_j1_a[1] == 1) or (pod_j1_b[0] == self.nb_tour and pod_j1_b[1] == 1): # (pod_j1_a[0] == nb_tour and pod_j1_a[1] == 1) car pour finir la carte, il faut revenir jusqu'au point de départ (CP dindice 0)
                self.pods = (1, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            elif parametre.defaite_j1_possible and (pod_j1_a[7]>parametre.nombre_de_tick_max_sans_cp and pod_j1_b[7]>parametre.nombre_de_tick_max_sans_cp) or (pod_j2_a[0] == self.nb_tour and pod_j2_a[1] == 1) or (pod_j2_b[0] == self.nb_tour and pod_j2_b[1] == 1):
                self.pods = (2, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            else:
                self.pods = (0, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            
            l = [self.pods[1].copy(), self.pods[2].copy(), self.pods[3].copy(), self.pods[4].copy()]
            self.memoire.append(l)

        self.premier_tour = False
    
        return self.get_observation(indice = 0), self.get_observation(indice = 2), self.get_reward_atq(rebond_fratricide), self.get_reward_dfs(rebond_fratricide, rebond_ennemi), (fin(self.pods) or self.tick>parametre.nombre_de_tick_max)
        # version 1 : 
        # return (self.get_observation(indice = 0), self.get_observation(indice = 1)), (self.get_observation(indice = 2), self.get_observation(indice = 3)), self.get_reward_atq(rebond_fratricide), self.get_reward_dfs(rebond_fratricide, rebond_ennemi), (fin(self.pods) or self.tick>parametre.nombre_de_tick_max)

    def get_reward_old(self, player):
        if len(self.memoire) <= 1:
            raise ValueError('No reward because not any action has been made')
        else:
            if self.memoire[-1][player][1] != self.memoire[-2][player][1]:
                return 1500
            else:
                distm1, distm2 = [sqrt((self.memoire[-i][player][2]-self.carte_cp[self.memoire[-i][player][1]][0])**2 + (self.memoire[-i][player][3]-self.carte_cp[self.memoire[-i][player][1]][1])**2) for i in (1, 2)]
                return (distm2-distm1)
    
    def get_reward(self, player):
        if len(self.memoire) <= 1:
            raise ValueError('No reward because not any action has been made')
        else:
            if player==0:
                return self.ecart_pods(0, 2)/500
            else:
                return self.ecart_pods(1, 3)/500
            
    def get_reward_atq(self, rebond_fratricide):
        var_dist_cp = self.get_reward(player = 0)
        if len(self.memoire) <= 1:
            raise ValueError('No reward because not any action has been made')
        else:
            var_dist = max([norme(self.memoire[-i][0][2]-self.memoire[-i][0][2], self.memoire[-i-1][0][3]-self.memoire[-i-1][0][3]) for i in range(min(5, len(self.memoire)-1))])
        
        return var_dist_cp/500 #+ min(0, (var_dist-100)*3/100) - (1 if rebond_fratricide else 0)

    def get_reward_dfs(self, rebond_fratricide, rebond_ennemi):
        # xd_hero, yd_hero, xa_adv, ya_adv = self.memoire[-1][1][2], self.memoire[-1][1][3], self.memoire[-1][2][2], self.memoire[-1][2][3]        
        # cp_0_xa, cp_0_ya = next_item(self.carte_cp, self.memoire[-1][2][1], 0)
        # x_bar, y_bar = self.barycentre
        
        # distance_cp, orientation_de_referance = norme_angle((xa_adv, ya_adv),(cp_0_xa,cp_0_ya), 0)

        # dist_adv, ang = norme_angle((xa_adv, ya_adv), (xd_hero, yd_hero), orientation_de_referance)

        # dist_barycentre = norme(xd_hero-x_bar, yd_hero-y_bar)

        # if distance_cp==0:
        #     facteur_prox_dist = 0
        # else:
        #     ratio_distance = dist_adv/distance_cp
        #     # facteur_prox_dist vaut 1 si dist est dans [cp_dist*0.1,cp_dist*0.9] et 0 en dehors de [0, cp_dist]
        #     # donc 1 si abs(ratio_distance-0.5)<=0.4 et 0 si abs(ratio_distance-0.5)>0.5
        #     facteur_prox_dist = min(1, max(0, (abs(ratio_distance-0.5)-0.5)*(-5))) 

        # facteur_prox_ang = min(1, max(0, (abs(ang)-45)/30)) # vaut 0 si ang est en dehors du cone de 45deg, 1 s'il est dan sle cone de 15 deg

        # reward_recul_adverse = -self.get_reward(player = 2)
        
        # downward_distance_barycentre = min(0, 1-exp((dist_barycentre-1000)/2000))
        # return facteur_prox_dist*facteur_prox_ang + (1 if rebond_ennemi else 0)

        var_dist_cp = self.get_reward(player = 2)
        return -var_dist_cp/500


    def get_n_cp(self, joueur = 0):
        return len(self.carte_cp)* self.memoire[-1][joueur][0] + self.memoire[-1][joueur][1] -1

    def get_observation_version1(self, indice):
        # pour chaque pods allié : 
        # -> vitesse (angle relatif a notre orientation, amplitude)
        # -> inactivité à cause du shield
        # -> boost encore disponible ? 

        # autres pod dans l'ordre : allié, adversaire attanquant, adversaire defenseur 
        # -> la position des autres pods (angle relatif a notre orientation, distance), 
        # -> la vitesse des autres pods (angle relatif a notre orientation)
        
        # puis pour le pod allié d'attaque : 
        # -> la position du next CP, et du next CP+1

        # puis pour le pod allié de defense : 
        # -> la position du next CP, du next CP+1, next CP+2 pour le pod d'attaque adverse relativement a notre pod 
        # -> la position du next CP, du next CP+1, next CP+2 pour le pod d'attaque adverse relativement au pod d'attaque adverse

        pod_a = self.memoire[-1][indice]

        if indice in [0, 2]: # pod d'attaque
            cp_indice = pod_a[1]
        else:
            if indice == 1: # pod de defense de l'equipe 1
                cp_indice = self.memoire[-1][2][1]
            else:
                cp_indice = self.memoire[-1][0][1]


        carte = self.carte_cp

        orientation_reference = pod_a[6]
        xa, ya = pod_a[2], pod_a[3]

        cp_0_xa, cp_0_ya = next_item(carte, cp_indice, 0)
        cp_1_xa, cp_1_ya = next_item(carte, cp_indice, 1)
        cp_2_xa, cp_2_ya = next_item(carte, cp_indice, 2)

        dist_cp_0, ang_cp_0 = norme_angle((xa, ya), (cp_0_xa, cp_0_ya), orientation_reference)
        dist_cp_1, ang_cp_1 = norme_angle((xa, ya), (cp_1_xa, cp_1_ya), orientation_reference)
        dist_cp_2, ang_cp_2 = norme_angle((xa, ya), (cp_2_xa, cp_2_ya), orientation_reference)


        dist_autres_pods, angle_autres_pods = [0, 0, 0], [0, 0, 0]
        norme_vitesse_pods, angle_vitesse_pods = [0, 0, 0, 0], [0, 0, 0, 0]
        
        vxa, vya = pod_a[4], pod_a[5]
        norme_vitesse_pods[0], angle_vitesse_pods[0] = norme_angle((0, 0), (vxa, vya), orientation_reference)


        liste_de_liste_de_pods = [
            [1, 2, 3],
            [0, 3, 2],
            [3, 0, 1],
            [2, 1, 0]
            ] # format : pod allié, pod de meme role, pod de role opposé
        
        liste_pods = [self.memoire[-1][i] for i in liste_de_liste_de_pods[indice]]

        for ind, pod in enumerate(liste_de_liste_de_pods[indice]):
            xp,yp, vxp, vyp = pod_a[2], pod_a[3], pod_a[4], pod_a[5]
            norme_vitesse_pods[ind+1], angle_vitesse_pods[ind+1] = norme_angle((0, 0), (vxp, vyp), orientation_reference)
            dist_autres_pods[ind], angle_autres_pods[ind] = norme_angle((xa, ya), (xp, yp), orientation_reference)

        l = [norm_vit(norme_vitesse_pods[j]) for j in range(4)
        ] + [norm_ang(angle_vitesse_pods[j]) for j in range(4)
        ] + [norm_dist(dist_autres_pods[j]) for j in range(3)
        ] + [norm_ang(angle_autres_pods[j]) for j in range(3)
        ] + [
            norm_dist(dist_cp_0),
            norm_ang(ang_cp_0),

            norm_dist(dist_cp_1),
            norm_ang(ang_cp_1),

            norm_dist(dist_cp_2),
            norm_ang(ang_cp_2)
            ]
        return l

    def get_observation(self, indice):
        if indice ==0:
            order_pods = [0, 1, 2, 3]
        elif indice ==2:
            order_pods = [2, 3, 0, 1]
        else:
            raise ValueError("l'indice de la fonction get_observation ne correspond à rien")
                
        lx = []
        ly = []
        lang = []
        lautre = []

        for ind_obs in range(2, 6):
            for ind_pod in order_pods:
                if ind_obs%2==0:
                    lx.append(self.memoire[-1][ind_pod][ind_obs])
                else:
                    ly.append(self.memoire[-1][ind_pod][ind_obs])

    
        for ind_pod in [order_pods[0], order_pods[2]]:
            indice_cp = self.memoire[-1][ind_pod][1]
            for depl in range(5):
                x, y = next_item(self.carte_cp, indice_cp, depl)
                lx.append(x)
                ly.append(y)


        for ind_pod in order_pods:
            lang.append(self.memoire[-1][ind_pod][6]/180 * pi)

        bit_fin, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2 = self.pods


        for ind_pod in order_pods:
            if ind_pod == 0:
                lautre.append(float(boost_j1))
                lautre.append(len(self.carte_cp)*(self.nb_tour-self.memoire[-1][ind_pod][0]) + self.memoire[-1][ind_pod][1])
                lautre.append((self.memoire[-1][ind_pod][7]-100)/20)
            if ind_pod == 2:
                lautre.append(float(boost_j2))
                lautre.append(len(self.carte_cp)*(self.nb_tour-self.memoire[-1][ind_pod][0]) + self.memoire[-1][ind_pod][1])
                lautre.append((self.memoire[-1][ind_pod][7]-100)/20)
    
        for ind_pod in order_pods:
            lautre.append(self.memoire[-1][ind_pod][8])
        
        
        return lx + ly + lang + lautre

    def ecart_pods(self, indice_pod_1, indice_pod_2):
        nombre_cp=len(self.carte_cp)
        
        indice_cp_1 = nombre_cp*self.memoire[-1][indice_pod_1][0] +self.memoire[-1][indice_pod_1][1]
        indice_cp_2 = nombre_cp*self.memoire[-1][indice_pod_2][0] +self.memoire[-1][indice_pod_2][1]
        
        if indice_cp_1>= indice_cp_2:
            
            distance = 0
            for k in range(indice_cp_2, indice_cp_1):
                distance += self.get_distance_cp(k%nombre_cp, (k+1)%nombre_cp)

            indice_cp_1=indice_cp_1%nombre_cp
            indice_cp_2=indice_cp_2%nombre_cp
            # distance du pod 2 à son cp
            distance += sqrt((self.memoire[-1][indice_pod_2][2] - self.carte_cp[indice_cp_2][0])**2 + (self.memoire[-1][indice_pod_2][3] - self.carte_cp[indice_cp_2][1])**2)
            # distance du pod 1 à son cp
            distance -= sqrt((self.memoire[-1][indice_pod_1][2] - self.carte_cp[indice_cp_1][0])**2 + (self.memoire[-1][indice_pod_1][3] - self.carte_cp[indice_cp_1][1])**2)
            return distance
        else:
            return -self.ecart_pods(indice_pod_2, indice_pod_1)


    def afficher(self):
        exemple = self.memoire
        lpod1 = []
        for _, l in enumerate(exemple):
            lpod1.append([[int(l[i][2]), int(l[i][3]), l[i][6]/180*pi]for i in range(4)])
        carte_cp_reduite = []

        for x,y in self.carte_cp:
            carte_cp_reduite.append((x, y))
        affgame(carte_cp_reduite, lpod1)
        
