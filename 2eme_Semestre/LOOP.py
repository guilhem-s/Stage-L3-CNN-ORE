import runpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--hidden", type=int, help="hidden size of the network")
parser.add_argument("-b", "--ethnie", type=str, help="ethnie reduced")
parser.add_argument("-c", "--proportion", type=int, help="reduction proportion")
parser.add_argument("-d", "--epoques", type=int, help="nombre d'epochs")


nb_cc = [50, 75, 100] #[100] #
proportion_test = [0, 10, 20, 30, 40, 50] # [50]#
ethnie = ["ch", "cau"]
epochs = 40 #[20, 50, 100, 150, 200]
#batch_size = [10, 20, 30, 40]
#fonction_act = ['relu', 'sigmoid', 'tanh']

for neurones in nb_cc:
    for eth in ethnie:
        for prop in proportion_test:
            for y in range(20):
                arguments = f'-a {neurones} -b {eth} -c {prop} -d {epochs}'
                args = parser.parse_args(arguments.split())

                # Call the other script using run_path and pass the parsed arguments
                runpy.run_path("C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre/ID_ORE_CNN.py", run_name="__main__", init_globals={'args': args})

               
        proportion_test.reverse()
        if 50 in proportion_test:
            proportion_test.remove(50)
        else:
            proportion_test.append(50)