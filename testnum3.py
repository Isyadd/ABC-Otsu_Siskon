import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import data

class ABC:
    def __init__(self, obj_function, lower_bound, upper_bound, colony_size=30, max_cycles=100, limit=20):
        self.obj_function = obj_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.colony_size = colony_size
        self.max_cycles = max_cycles
        self.limit = limit
        self.dim = 1  # Dimension of the problem (threshold value)
        
    def optimize(self):
        # Initialize the population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.colony_size, self.dim))
        fitness = np.array([self.obj_function(ind) for ind in population])
        trial = np.zeros(self.colony_size)
        
        for cycle in range(self.max_cycles):
            # Employed bees phase
            for i in range(self.colony_size):
                new_solution = self.mutate(population, i)
                new_fitness = self.obj_function(new_solution)
                if new_fitness > fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

            # Onlooker bees phase
            probabilities = fitness / fitness.sum()
            for i in range(self.colony_size):
                selected_index = np.random.choice(np.arange(self.colony_size), p=probabilities)
                new_solution = self.mutate(population, selected_index)
                new_fitness = self.obj_function(new_solution)
                if new_fitness > fitness[selected_index]:
                    population[selected_index] = new_solution
                    fitness[selected_index] = new_fitness
                    trial[selected_index] = 0
                else:
                    trial[selected_index] += 1

            # Scout bees phase
            for i in range(self.colony_size):
                if trial[i] > self.limit:
                    population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[i] = self.obj_function(population[i])
                    trial[i] = 0
        
        # Return the best solution found
        best_index = np.argmax(fitness)
        return population[best_index], fitness[best_index]

    def mutate(self, population, index):
        phi = np.random.uniform(-1, 1, self.dim)
        partner_index = np.random.choice(np.delete(np.arange(self.colony_size), index))
        new_solution = population[index] + phi * (population[index] - population[partner_index])
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        return new_solution

def between_class_variance(threshold, hist, total_pixels):
    threshold = int(threshold)
    weight_background = np.sum(hist[:threshold]) / total_pixels
    weight_foreground = 1 - weight_background
    
    mean_background = np.sum(np.arange(threshold) * hist[:threshold]) / (weight_background * total_pixels + 1e-10)
    mean_foreground = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (weight_foreground * total_pixels + 1e-10)
    
    variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    return variance_between

def otsu_abc_threshold(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    
    def obj_function(threshold):
        return between_class_variance(threshold, hist, total_pixels)
    
    abc = ABC(obj_function, lower_bound=0, upper_bound=255, colony_size=30, max_cycles=100, limit=20)
    optimal_threshold, _ = abc.optimize()
    optimal_threshold = int(optimal_threshold)
    
    otsu_thresholded = (image < optimal_threshold).astype(np.uint8) * 255
    
    return optimal_threshold, otsu_thresholded, hist

# Usage example
if __name__ == "__main__":
    # Load the image in grayscale
    # image = cv2.imread('', cv2.IMREAD_GRAYSCALE)
    image = data.camera() 

    # Check if the image was successfully loaded
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Apply Gaussian blur to the image to reduce noise (optional but recommended)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform Otsu's thresholding using the ABC algorithm
    optimal_threshold, otsu_thresholded, hist = otsu_abc_threshold(blurred_image)

    print(f"Optimal threshold value determined by ABC algorithm: {optimal_threshold}")

    # Display the original, thresholded images, and histogram using matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('ABC-Otsu Thresholding')
    plt.imshow(otsu_thresholded, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Histogram')
    plt.plot(hist, color='black')
    plt.axvline(x=optimal_threshold, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
