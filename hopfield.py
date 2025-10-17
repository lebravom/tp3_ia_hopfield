import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    def __init__(self, size=10):
        """
        Red de Hopfield para detección de patrones
        size: tamaño de la matriz (size x size)
        """
        self.size = size
        self.n_neurons = size * size
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        self.patterns = []
        self.method = None
        
    def train_hebb(self, patterns):
        """
        Entrenamiento usando la Regla de Hebb
        W = (1/P) * sum(p_i * p_i^T)
        """
        self.method = "Hebb"
        self.patterns = patterns
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        P = len(patterns)
        for pattern in patterns:
            p = pattern.flatten()
            self.weights += np.outer(p, p)
        
        self.weights /= P
        np.fill_diagonal(self.weights, 0)
        
        print(f"=== Entrenamiento con Regla de Hebb ===")
        print(f"Patrones almacenados: {P}")
        print(f"Dimensión de W: {self.weights.shape}")
        
    def train_pseudoinverse(self, patterns):
        """
        Entrenamiento usando Matriz Pseudoinversa
        W = X * X^+
        donde X es la matriz de patrones (cada columna es un patrón)
        y X^+ es la pseudoinversa de Moore-Penrose
        """
        self.method = "Pseudoinversa"
        self.patterns = patterns
        
        # Crear matriz X (cada columna es un patrón)
        n_patterns = len(patterns)
        X = np.zeros((self.n_neurons, n_patterns))
        for i, pattern in enumerate(patterns):
            X[:, i] = pattern.flatten()
        
        # Calcular pseudoinversa: W = X * X^+
        # donde X^+ es la pseudoinversa de Moore-Penrose de X
        try:
            X_pinv = np.linalg.pinv(X)  # X^+ tiene dimensión (n_patterns x n_neurons)
            self.weights = X @ X_pinv    # (n_neurons x n_patterns) @ (n_patterns x n_neurons)
            np.fill_diagonal(self.weights, 0)
            
            print(f"=== Entrenamiento con Matriz Pseudoinversa ===")
            print(f"Patrones almacenados: {n_patterns}")
            print(f"Dimensión de X: {X.shape}")
            print(f"Dimensión de X^+: {X_pinv.shape}")
            print(f"Dimensión de W: {self.weights.shape}")
            print(f"Rango de la matriz de patrones: {np.linalg.matrix_rank(X)}")
        except np.linalg.LinAlgError:
            print("Error: No se pudo calcular la pseudoinversa")
            self.weights = np.zeros((self.n_neurons, self.n_neurons))
    
    def recall(self, pattern, max_iter=100, verbose=False):
        """
        Recuperación del patrón mediante actualización asíncrona
        """
        state = pattern.flatten().copy()
        history = [state.copy().reshape(self.size, self.size)]
        
        for iteration in range(max_iter):
            prev_state = state.copy()
            
            # Actualización asíncrona secuencial
            for i in range(self.n_neurons):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
            
            history.append(state.copy().reshape(self.size, self.size))
            
            # Verificar convergencia
            if np.array_equal(state, prev_state):
                if verbose:
                    print(f"Convergió en iteración {iteration + 1}")
                break
        else:
            if verbose:
                print(f"No convergió en {max_iter} iteraciones")
        
        return state.reshape(self.size, self.size), history
    
    def energy(self, pattern):
        """Función de energía de Hopfield: E = -0.5 * s^T * W * s"""
        state = pattern.flatten()
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def calculate_overlap(self, state, pattern):
        """Calcula el solapamiento entre dos patrones (medida de similitud)"""
        s1 = state.flatten()
        s2 = pattern.flatten()
        return np.dot(s1, s2) / self.n_neurons


def create_reference_mark(size=10):
    """
    Crea una marca de referencia (cruz en esquina superior izquierda)
    Esta marca es inalterable y sirve como referencia de posición
    """
    mark = np.ones((size, size)) * -1
    # Cruz pequeña en esquina
    mark[0:2, 0] = 1  # Vertical
    mark[0, 0:2] = 1  # Horizontal
    return mark


def create_circle(size, center, radius):
    """Crea un círculo"""
    matrix = create_reference_mark(size)
    cx, cy = center
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if abs(dist - radius) < 1.0:
                matrix[i, j] = 1
    
    return matrix

def add_noise(pattern, noise_level):
    """Añade ruido volteando píxeles aleatoriamente"""
    noisy = pattern.copy()
    n_flips = int(100 * noise_level)
    
    # Evitar voltear la marca de referencia
    indices = []
    for idx in range(100):
        i, j = idx // 10, idx % 10
        if not (i < 2 and j < 2):  # No tocar la marca de referencia
            indices.append(idx)
    
    flip_indices = np.random.choice(indices, min(n_flips, len(indices)), replace=False)
    
    for idx in flip_indices:
        i, j = idx // 10, idx % 10
        noisy[i, j] *= -1
    
    return noisy


def print_pattern(pattern, title=""):
    """Imprime el patrón usando caracteres"""
    if title:
        print(f"\n{title}")
        print("=" * 22)
    
    print("  ", end="")
    for j in range(10):
        print(j, end=" ")
    print()
    
    for i in range(10):
        print(f"{i} ", end="")
        for j in range(10):
            if pattern[i, j] == 1:
                print("█", end=" ")
            else:
                print("·", end=" ")
        print()


def find_center(pattern):
    """Encuentra el centro de masa de la figura (excluyendo marca de referencia)"""
    coords_y, coords_x = np.where(pattern == 1)
    
    # Filtrar la marca de referencia
    valid_coords = [(y, x) for y, x in zip(coords_y, coords_x) if not (y < 2 and x < 2)]
    
    if not valid_coords:
        return None, None
    
    center_y = np.mean([c[0] for c in valid_coords])
    center_x = np.mean([c[1] for c in valid_coords])
    
    return center_x, center_y


def run_experiment(network, patterns, pattern_names, test_pattern, test_name, noise_level):
    """Ejecuta un experimento completo"""
    print("\n" + "="*60)
    print(f"EXPERIMENTO: {test_name}")
    print(f"Método: {network.method}")
    print("="*60)
    
    # Patrón limpio
    print_pattern(test_pattern, "Patrón Original (sin ruido)")
    cx_orig, cy_orig = find_center(test_pattern)
    if cx_orig:
        print(f"Centro original: ({cx_orig:.2f}, {cy_orig:.2f})")
    
    # Añadir ruido
    noisy = add_noise(test_pattern, noise_level)
    print_pattern(noisy, f"Patrón con Ruido ({int(noise_level*100)}%)")
    
    n_different = np.sum(noisy != test_pattern)
    print(f"Píxeles modificados por ruido: {n_different}")
    
    # Recuperación
    recovered, history = network.recall(noisy, verbose=True)
    print_pattern(recovered, "Patrón Recuperado")
    
    # Análisis del resultado
    print("\n--- Análisis ---")
    cx_rec, cy_rec = find_center(recovered)
    if cx_rec:
        print(f"Centro recuperado: ({cx_rec:.2f}, {cy_rec:.2f})")
        if cx_orig:
            error = np.sqrt((cx_rec - cx_orig)**2 + (cy_rec - cy_orig)**2)
            print(f"Error en centro: {error:.2f} píxeles")
    
    # Solapamiento con patrones almacenados
    print("\nSolapamiento con patrones almacenados:")
    for i, (stored, name) in enumerate(zip(patterns, pattern_names)):
        overlap = network.calculate_overlap(recovered, stored)
        print(f"  {name}: {overlap:.3f}")
    
    # Energía
    E_noisy = network.energy(noisy)
    E_recovered = network.energy(recovered)
    print(f"\nEnergía con ruido: {E_noisy:.2f}")
    print(f"Energía recuperada: {E_recovered:.2f}")
    print(f"Reducción de energía: {E_noisy - E_recovered:.2f}")
    
    # Convergencia
    print(f"\nIteraciones hasta convergencia: {len(history) - 1}")
    
    return recovered, history


def main():
    """Función principal del prototipo"""
    print("="*60)
    print("PROTOTIPO: RED DE HOPFIELD 10x10")
    print("Detección de Figuras Geométricas")
    print("="*60)
    
    # Crear patrones base
    circle = create_circle(10, center=(6, 5), radius=3)
    circle1 = create_circle(10, center=(5, 5), radius=3)
    circle2 = create_circle(10, center=(4, 5), radius=3)

    patterns = [circle, circle1, circle2]
    pattern_names = ["Círculo","Círculo 2", "Círculo 3"]
    
    print("\n--- PATRONES ALMACENADOS ---")
    for pattern, name in zip(patterns, pattern_names):
        print_pattern(pattern, name)
    
    # ===== EXPERIMENTO 1: Regla de Hebb =====
    print("\n\n" + "#"*60)
    print("# PARTE 1: ENTRENAMIENTO CON REGLA DE HEBB")
    print("#"*60)
    
    net_hebb = HopfieldNetwork(size=10)
    net_hebb.train_hebb(patterns)
    
    # Pruebas con Hebb
    run_experiment(net_hebb, patterns, pattern_names, random.choice(patterns), "Círculo con ruido", 0.20)

    
    # ===== EXPERIMENTO 2: Pseudoinversa =====
    print("\n\n" + "#"*60)
    print("# PARTE 2: ENTRENAMIENTO CON PSEUDOINVERSA")
    print("#"*60)
    
    net_pinv = HopfieldNetwork(size=10)
    net_pinv.train_pseudoinverse(patterns)
    
    # Pruebas con Pseudoinversa
    run_experiment(net_pinv, patterns, pattern_names, random.choice(patterns), "Círculo con ruido", 0.20)

    
    # ===== COMPARACIÓN =====
    print("\n\n" + "#"*60)
    print("# COMPARACIÓN DE MÉTODOS")
    print("#"*60)
    
    test = add_noise(random.choice(patterns), 0.25)
    print_pattern(test, "Patrón de prueba (Círculo con 25% ruido)")
    
    rec_hebb, _ = net_hebb.recall(test)
    rec_pinv, _ = net_pinv.recall(test)
    
    print_pattern(rec_hebb, "Recuperado con Hebb")
    print(f"Energía: {net_hebb.energy(rec_hebb):.2f}")
    print(f"Solapamiento con círculo: {net_hebb.calculate_overlap(rec_hebb, circle):.3f}")
    
    print_pattern(rec_pinv, "Recuperado con Pseudoinversa")
    print(f"Energía: {net_pinv.energy(rec_pinv):.2f}")
    print(f"Solapamiento con círculo: {net_pinv.calculate_overlap(rec_pinv, circle):.3f}")
    

if __name__=="__main__":
    main()
