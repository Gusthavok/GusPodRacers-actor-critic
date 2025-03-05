import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic_v5(nn.Module):

    def __init__(self,  _a, _b):
        super(Critic_v5, self).__init__()

        n_ang, n_points = 32, 32
        
        self.combinaison_points = nn.Linear(14*4 + 4, n_points) # 14 position fois 4 centre de projection possible, plus 4 vitesses
        self.norm_vitesse = 200
        self.norm_distance = 2000
        # self.combinaisons_angles = nn.Linear(4+n_points, n_ang)

        self.finalMLP = nn.ModuleList([nn.Linear(10+n_points+2*(n_ang*4) + 8, 32)] + [nn.Linear(32, 32) for _ in range(1)])

        self.final_layer = nn.Linear(32, 2)

    
    def forward(self, action, observation):
        x_pods, vit_x_pods, x_autre, y_pods, vit_y_pods, y_autre, lang, lautre = observation[:, :4], observation[:, 4:8], observation[:, 8:18], observation[:, 18:22], observation[:, 22:26], observation[:, 26:36], observation[:, 36:40], observation[:, 40:50]
        
        x_tot = torch.cat((x_pods, x_autre), dim=1)
        y_tot = torch.cat((y_pods, y_autre), dim=1)
        
        lx = torch.cat([(x_tot - x_pods[:, i:i+1])/self.norm_distance for i in range(4)] + [vit_x_pods/self.norm_vitesse], dim=1)
        ly = torch.cat([(y_tot - y_pods[:, i:i+1])/self.norm_distance for i in range(4)] + [vit_y_pods/self.norm_vitesse], dim=1)

        lx = self.combinaison_points(lx)
        ly = self.combinaison_points(ly)

        ang_from_xy = torch.atan2(ly, lx)

        angs = torch.cat([ang_from_xy - lang[:, i:i+1] for i in range(4)], dim=1)
        # angs = self.combinaisons_angles(angs)
        lcos, lsin = torch.cos(angs), torch.sin(angs)

        lnormes = torch.sqrt(lx*lx + ly*ly)

        x = torch.cat((lautre, lnormes, lcos, lsin, action), dim=1)

        for layer in self.finalMLP:
            x = F.leaky_relu(layer(x))

        return self.final_layer(x)