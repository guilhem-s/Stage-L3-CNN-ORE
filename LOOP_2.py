import runpy

nb_cc = [2, 6, 8]
proportion_test = [0, 10, 20, 30, 40, 50]
ethnie = [0, 1]

for neurones in nb_cc:
    for eth in ethnie:
        for prop in proportion_test:
            for y in range(1):    #car l'argument d ne fait pas 10 run lorsque sa valeur vaut 10
                arguments = '-a '+str(neurones)+' -b '+str(eth)+' -c '+str(prop)+' -d '+str(1)
                runfile("C:/Users/Guilem/Stage/CNN_ORE/ID_CAT.py", args=arguments, wdir="C:/Users/Guilem/Stage/CNN_ORE")                
                # print(neurones, eth, prop)
                
        proportion_test.reverse()
        if 50 in proportion_test:
            proportion_test.remove(50)
        else:
            proportion_test.append(50)