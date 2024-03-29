import runpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--hidden", type=int, help="hidden size of the network")
parser.add_argument("-b", "--ethnie", type=str, help="ethnie reduced")
parser.add_argument("-c", "--proportion", type=int, help="reduction proportion")
parser.add_argument("-d", "--run", type=int, help="nombre d'epochs")

nb_cc = [10] #[10, 20, 30, 40, 50, 75, 100]
proportion_test = [40] #[0, 10, 20, 30, 40, 50]
ethnie = ["ch"]
epochs = 100

for neurones in nb_cc:
    for eth in ethnie:
        for prop in proportion_test:
            for y in range(5):
                arguments = f'-a {neurones} -b {eth} -c {prop} -d {y}'
                args = parser.parse_args(arguments.split())

                # Call the other script using run_path and pass the parsed arguments
                runpy.run_path("C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre/ID_CNN_ORE-Revu.py", run_name="__main__", init_globals={'args': args})

               
        """ proportion_test.reverse()
        if 50 in proportion_test:
            proportion_test.remove(50)
        else:
            proportion_test.append(50) """