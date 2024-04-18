import re
from .atomic_units import AtomicUnits as au
_separators=" |,|=|\t|\n"
_comment_chars=["#","!"]
_true_repr=["true","yes",".true."]
_false_repr=["false","no",".false."]

class InputFile(dict):
	case_insensitive=True
	def get(self, path, default = None):
		if InputFile.case_insensitive:
			path=path.lower()
		keys = path.split("/")
		val = None
		for key in keys:
			if isinstance(val,InputFile):
				val = val.get(key, default=None)
			else:
				val = dict.get(self, key, None)

			if val is None:
				return default
		
		return val

	def store(self, path, value):
		if InputFile.case_insensitive:
			path=path.lower()
		keys = path.split("/")
		child = self.get(keys[0],default=None)
		if isinstance(child,InputFile):
			if len(keys) == 1:
				print("Warning: overriding a sub-dictionary!")
				self[keys[0]]=value
				return 1
			else:
				child.store("/".join(keys[1:]),value)
		else:
			if len(keys) == 1:
				self[keys[0]]=value
				return 0
			else:
				if child is None:
					self[keys[0]]=InputFile()
					self[keys[0]].store("/".join(keys[1:]),value)
				else:
					print("Error: hit a leaf before the end of path!")
					return -1

	
	def print(self,tab=""):
		string = ""
		for p_id, p_info in self.items():
			string += tab+p_id
			val=self.get(p_id)
			if isinstance(val,InputFile):
				string += "{\n" + val.print(tab=tab+"  ") +"\n"+tab+"}\n\n"
			else:
				string += " = "+str(val)+"\n"			
		return string[:-1]
	
	def save(self,filename):
		with open(filename,"w") as f:
			f.write(self.print())
	
	def __str__(self):
		return self.print()


def parse_input(input_file):
# parse an input file and return a nested dictionary
# containing the categories, keys and values
	f=open(input_file,'r')
	struct=InputFile()
	path=[]
	for line in f:
		#remove all after comment character
		for comment_char in _comment_chars:
			index=line.find(comment_char)
			if index >= 0:
				line=line[:index]

		#split line using defined separators
		parsed_line=re.split(_separators,
						line.strip())

		#remove empty strings
		parsed_line=[x for x in parsed_line if x]
		#skip blank lines
		if not parsed_line:
			continue
		#print(parsed_line)

		word0 = parsed_line[0].lower()
		cat_fields=''.join(parsed_line)
		#check if beginning of a category
		if cat_fields.endswith("{"):
			path.append(cat_fields[:-1])
			continue
		if cat_fields.startswith("&"):
			path.append(cat_fields[1:])
			continue
		if cat_fields.endswith("{}"):
			struct.store("/".join("path")+"/"+cat_fields[1:-2]
							,InputFile())
			continue		
		
		#print(current_category)
		#if not path:
		#	print("Error: line not recognized!")
		#	return None
		# else: #check if end of a category
		if (cat_fields[0] in "}/") or ("&end" in cat_fields):
			del path[-1]
			continue
		
		word0, unit=_get_unit_from_key(word0)
		val=None
		if len(parsed_line) == 1:
			val=True  # keyword only => store True
		elif len(parsed_line) == 2:
			val=string_to_true_type(parsed_line[1],unit)
		else:
			#analyze parsed line
			val=[]
			for word in parsed_line[1:]:
				val.append(string_to_true_type(word,unit))
		struct.store("/".join(path+[word0]),val)

	f.close()
	return struct

def string_to_true_type(word,unit=None):
	if unit is not None:
		return float(word)/unit
	
	try:
		val=int(word)
	except ValueError:
		try:
			val=float(word)
		except ValueError:
			if word.lower() in _true_repr:
				val=True
			elif word.lower() in _false_repr:
				val=False
			else:
				val=word
	return val

def _get_unit_from_key(word):
	unit_start=max(word.find("{"),word.find("["))
	n=len(word)
	if unit_start<0:
		key=word
		unit=None
	elif unit_start==0:
		print("Error: Field '"+str(word)+"' must not start with '{' or '[' !")
		raise ValueError
	else:
		if word[unit_start] == "{":
			end_bracket="}"
		else:
			end_bracket="]"
		key=word[:unit_start]
		if word[n-1] != end_bracket:
			print("Error: wrong unit specification in field '"+str(word)+"' !")
			raise ValueError
		
		if n-unit_start-2 < 0 :
			unit=1.
		else :
			unit=au.get_multiplier(word[unit_start+1:-1])
			#print(key+" unit= "+str(unit))
	return key, unit
