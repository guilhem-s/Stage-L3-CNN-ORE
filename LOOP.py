# -*- coding: utf-8 -*-
import runpy

nb_cc = [0, 4, 8]
proportion_test = [0, 10, 20, 30, 40, 50]
ethnie_2 = [0, 1]

for cc in nb_cc:
    for eth in ethnie_2:
        for prop in proportion_test:
            for y in range(10):  
                arguments = '-a '+str(cc)+' -b '+str(eth)+' -c '+str(prop)+' -d '+str(1)
                runfile("C:/Users/Guilem/Stage/CNN_ORE/CNN_ORE_CAT.py", args=arguments, wdir="C:/Users/Guilem/Stage/CNN_ORE")
                # print(cc, eth, prop)
        proportion_test.reverse()
        if 50 in proportion_test:
            proportion_test.remove(50)
        else:
            proportion_test.append(50)