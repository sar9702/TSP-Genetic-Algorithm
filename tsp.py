import random
import numpy as np

POPULATION_SIZE = 10  # 염색체 집합의 크기
SIZE = 4  # 유전자 개수
ITERATIONS = 100  # 세대 수
MUTATION_RATE = 0.2  # 돌연변이 확률

# 거리 정보
distances = {
    "A": {"B": 10, "C": 15, "D": 20},
    "B": {"A": 10, "C": 25, "D": 35},
    "C": {"A": 15, "B": 25, "D": 18},
    "D": {"A": 20, "B": 35, "C": 18},
}

class Chromosome:
    def __init__(self, genes = []):
        self.genes = genes
        self.fitness = self.cal_fitness()
    
    def cal_fitness(self):
        total_distance = 0
        for i in range(len(self.genes) - 1):
            total_distance += distances[self.genes[i]][self.genes[i + 1]]
        total_distance += distances[self.genes[len(self.genes) - 1]][self.genes[0]]
        return total_distance

def select(population):
    '''
    룰렛 휠 방법으로 염색체를 선택하여 반환하는 함수
    '''
    max = sum([p.fitness for p in population])
    selection_probs = [p.fitness/max for p in population]
    selected = population[np.random.choice(len(population), p=selection_probs)]
    return selected

def crossover(population):
    '''
    염색체를 선택하고 교차하여 자손을 생성하는 함수
    '''
    father = select(population)
    mother = select(population)
    length = random.randint(1, SIZE - 1)
    index = random.randint(0, SIZE - length)

    t_child1 = father.genes[index:index + length].copy()
    t_child2 = mother.genes[index:index + length].copy()

    child1 = list(filter(lambda x: not x in t_child1, father.genes))
    child2 = list(filter(lambda x: not x in t_child2, mother.genes))

    child1 = child1[:index] + t_child1 + child1[index:]
    child2 = child2[:index] + t_child2 + child2[index:]

    return child1, child2

def mutate(p):
    '''
    돌연변이를 생성하는 함수
    '''
    if random.random() < MUTATION_RATE:
        x, y = random.sample(list(range(0, SIZE)), 2)
        p.genes[y], p.genes[x] = p.genes[x], p.genes[y]

if __name__ == "__main__":
    # 초기 염색체 집합
    initial_genes = ["D", "C", "B", "A"]
    population = [Chromosome(random.sample(initial_genes, len(initial_genes))) for _ in range(POPULATION_SIZE)]

    output = []
    for i in range(ITERATIONS):
        new_population = []
        for _ in range(len(population) // 2):
            child1, child2 = crossover(population)
            new_population.append(Chromosome(child1))
            new_population.append(Chromosome(child2))
        
        population = new_population.copy()

        for p in population:
            mutate(p)

        # 출력을 위한 정렬
        population.sort(key=lambda x: x.fitness)
        output.append(population[0])
        print("{} 세대 : {} / 적합도 = {}".format(i + 1, population[0].genes, population[0].fitness))

    output.sort(key=lambda x: x.fitness)
    print("Best path: ", output[0].genes)
    print("Distance: ", output[0].fitness)
