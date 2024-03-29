import argparse

DEF_SRC_FILE = "./222.txt"

def main(src_file):
	
	with open (src_file) as f:

		part_dims = next(f)

		mt, kt, nt = ( int(x) for x in part_dims.split())

		counts = [mt * kt, kt * nt, mt * nt]
		lines = []

		for line in f:
			if line[0] == "#": continue
			lines.append([ int(x) for x in (line.strip().split()) ])

		R = len(lines[0])

		U = lines[0 : counts[0]]
		V = lines[counts[0]: counts[0] + counts[1]]
		W = lines[counts[0] + counts[1] : ]

		print(f"const int M_TILDE = {mt};\nconst int N_TILDE = {nt};\nconst int K_TILDE = {kt};\n")

		ustr = str(U).replace("[", "{").replace("]", "}")
		vstr = str(V).replace("[", "{").replace("]", "}")
		wstr = str(W).replace("[", "{").replace("]", "}")

		print(f"int U[{counts[0]}][{R}] = {ustr};\n")
		print(f"int V[{counts[1]}][{R}] = {vstr};\n")
		print(f"int W[{counts[2]}][{R}] = {wstr};\n")


if __name__ == "__main__":

	import argparse
 
	msg = "Adding description"
	 
	# Initialize parser
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", default=DEF_SRC_FILE)
	args = parser.parse_args()
	main(args.s)