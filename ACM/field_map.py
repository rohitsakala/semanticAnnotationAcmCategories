#!/usr/bin/python

import random
import sys
import math

#filename = raw_input("Enter the file name\n")
filename = sys.argv[1]

try: 

	f = open(filename,'r') 
except IOError: 
	print 'unable to open the file.'
	sys.exit(1)

class_dict = {}

for line in f:

	line.rstrip()
	if line[0] == '#' and line[1] == 'f':
		if ~( class_dict.has_key(line) ):
			class_dict.setdefault( line[2:-2], None)

f.close()

print class_dict.keys()

list_computational_science = ['scientific_computing', ]

list_ai = ['human-computer_interaction','real_time_and_embedded_systems','algorithms_and_theory','networks_and_communications','simulation','computer_education','machine_learning_and_pattern_recognition','natural_language_and_speech','artificial_intelligence']
list_ir = ['information_retrieval','world_wide_web','multimedia','distributed_and_parallel_computing']
list_db = ['security_and_privacy','data_mining','databases','programming_languages','hardware_and_architecture','software_engineering','operating_systems']
list_cv = ['computer_vision', 'graphics','bioinformatics_and_computational_biology']

temp_list = [ list_ai, list_ir, list_db, list_cv]

manual_class_dict = {}

for val in list_ai:
	manual_class_dict[val]='AI'

for val in list_ir:
	manual_class_dict[val]='IR'

for val in list_db:
	manual_class_dict[val]='DB'

for val in list_cv:
	manual_class_dict[val]='CV'

print "\n\nmanual classification into 4 categories\n\n"
print manual_class_dict
