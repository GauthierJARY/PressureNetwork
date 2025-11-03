import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Node:
    def __init__(self, name, volume=0, initial_pressure=101325):
        self.name = name
        self.volume = volume  # m³
        self.pressure = initial_pressure  # Pa
        self.connections = []
        self.mass = (initial_pressure * volume) / (287 * 300)  # kg (air)

    def add_connection(self, element):
        self.connections.append(element)

    def get_total_inflow(self):
        return sum([elem.get_flow(self) for elem in self.connections if elem.is_inflow(self)])

    def get_total_outflow(self):
        return sum([elem.get_flow(self) for elem in self.connections if elem.is_outflow(self)])

class Element:
    def __init__(self, name):
        self.name = name
        self.node1 = None
        self.node2 = None

    def connect(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_connection(self)
        node2.add_connection(self)

    def is_inflow(self, node):
        return self.node1 == node

    def is_outflow(self, node):
        return self.node2 == node

    def get_flow(self, node):
        raise NotImplementedError

class Pipe(Element):
    def __init__(self, name, diameter, length, roughness=0.0001):
        super().__init__(name)
        self.diameter = diameter  # m
        self.length = length  # m
        self.roughness = roughness  # m
        self.area = np.pi * (diameter/2)**2  # m²

    def get_flow(self, node):
        # Équation de Darcy-Weisbach
        if self.node1 == node:
            delta_P = self.node1.pressure - self.node2.pressure
        else:
            delta_P = self.node2.pressure - self.node1.pressure

        if delta_P <= 0:
            return 0

        # Calcul du coefficient de frottement (approximation)
        reynolds = 1e5  # Approximation - devrait être calculé en fonction du débit
        f = 0.001  # Coefficient de frottement approximatif

        # Équation de Darcy-Weisbach
        mass_flow = np.sqrt(2 * delta_P * self.area**2 / (f * self.length * 287 * 300)) * 3600
        return mass_flow

class Pump(Element):
    def __init__(self, name, performance_curve):
        super().__init__(name)
        self.performance_curve = performance_curve  # Fonction Q(P)

    def get_flow(self, node):
        # La pompe aspire du node1 vers node2
        if self.node1 == node:
            # Calcul du débit en fonction de la pression d'entrée
            inlet_pressure = self.node1.pressure
            return self.performance_curve(inlet_pressure)
        else:
            return -self.performance_curve(self.node2.pressure)

# Définition de la courbe de performance de la pompe
def pump_performance_curve(P_in):
    # Interpolation linéaire entre les points donnés
    points = np.array([
        [0.03, 0],    # Point bas pression
        [0.1, 33],    # Point milieu
        [1, 54],      # Point haut pression
        [10, 54],     # Point maximum
        [100, 20],    # Point descendant
        [1000, 8],    # Point bas débit
        [2000, 8],    # Point maximum pression
        [3, 55],      # Point supplémentaire
        [300, 10]     # Point supplémentaire
    ])

    # Tri des points par pression croissante
    points = points[points[:, 0].argsort()]

    # Interpolation linéaire
    from scipy.interpolate import interp1d
    curve = interp1d(points[:, 0], points[:, 1], kind='linear', fill_value="extrapolate")
    return curve(P_in)

# Création du système
tank = Node("Tank", volume=3, initial_pressure=101325)  # 3 m³, pression atmosphérique initiale
outlet = Node("Outlet", volume=0, initial_pressure=101325)  # Nœud de sortie

# Création des éléments
pump = Pump("Pump", pump_performance_curve)
pipe1 = Pipe("Pipe1", diameter=0.02, length=2)  # 2 cm de diamètre, 2 m de long
pipe2 = Pipe("Pipe2", diameter=0.02, length=1)  # 2 cm de diamètre, 1 m de long

# Connexion des éléments
pump.connect(tank, outlet)
pipe1.connect(tank, pump)
pipe2.connect(pump, outlet)

# Équation différentielle pour le système
def system_dynamics(y, t):
    # y = [tank.mass, tank.pressure, outlet.pressure]
    tank_mass, tank_pressure, outlet_pressure = y

    # Mise à jour des pressions
    tank.pressure = tank_pressure
    outlet.pressure = outlet_pressure

    # Calcul des débits
    m_dot_in = pipe1.get_flow(tank)  # Débit entrant dans la cuve
    m_dot_out = pump.get_flow(tank)  # Débit sortant de la cuve

    # Conservation de la masse dans la cuve
    dm_dt = m_dot_in - m_dot_out

    # Relation PV = nRT pour la cuve
    dP_dt = (287 * 300 / tank.volume) * dm_dt

    # Pour le nœud de sortie, on suppose la pression constante (atmosphérique)
    dP_out_dt = 0

    return [dm_dt, dP_dt, dP_out_dt]

# Conditions initiales
y0 = [tank.mass, tank.pressure, outlet.pressure]

# Temps de simulation
t = np.linspace(0, 1000, 1000)

# Résolution
sol = odeint(system_dynamics, y0, t)

# Extraction des résultats
tank_mass = sol[:, 0]
tank_pressure = sol[:, 1]
outlet_pressure = sol[:, 2]

# Tracé des résultats
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, tank_pressure/1000, label='Pression dans la cuve (kPa)')
plt.xlabel('Temps (s)')
plt.ylabel('Pression (kPa)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, tank_mass, label='Masse dans la cuve (kg)')
plt.xlabel('Temps (s)')
plt.ylabel('Masse (kg)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, outlet_pressure/1000, label='Pression de sortie (kPa)')
plt.xlabel('Temps (s)')
plt.ylabel('Pression (kPa)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
