import runpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--hidden", type=int, help="hidden size of the network")
parser.add_argument("-b", "--ethnie", type=str, help="ethnie reduced")
parser.add_argument("-c", "--proportion", type=int, help="reduction proportion")
parser.add_argument("-d", "--epoques", type=int, help="nombre d'epochs")


nb_cc = [10, 20, 30, 40, 50, 75, 100] #[100] #
proportion_test = [0, 10, 20, 30, 40, 50] # [50]#
ethnie = ["cau", "ch"]
epochs = 50 #[20, 50, 100, 150, 200]
#batch_size = [10, 20, 30, 40]
#fonction_act = ['relu', 'sigmoid', 'tanh']

for eth in ethnie:
    for prop in proportion_test:
        for y in range(1):
            print(eth, prop)
            
    proportion_test.reverse()
    if 50 in proportion_test:
        proportion_test.remove(50)
    else:
        proportion_test.append(50)