import subprocess

def main():

    outputs = dict()
    
    for m in range(240, 3500, 240):

        data = subprocess.run(f"./driver.x {m} {m} {m} -f 1 222.txt", capture_output=True, shell=True)

        output = data.stdout.splitlines()

        res_obj = {}

        for line in output:
            if "RESULT" in line: continue

            line = line.strip()
            sep_index = line.index(" ")
            key = line[0:sep_index]
            value = line[sep_index + 1:]

            res_obj[key] = value

        outputs[",".join([res_obj["m"], res_obj["n"], res_obj["k"]])] = res_obj

        # outputs[m] = {

        # }
        
        # pre_line_0 = output[12]
        # pre_line_1 = output[13]

        # adj_line_0 = output[17]
        # adj_line_1 = output[18]
        
        # pre_gflops, pre_time = get_line_0(pre_line_0)
        # pre_acc, pre_ukr, pre_packa, pre_packb = get_line_1(pre_line_1)

        # adj_gflops, adj_time = get_line_0(adj_line_0)
        # adj_acc, adj_ukr, adj_packa, adj_packb = get_line_1(adj_line_1)

        # outputs[m] = {
        #     "m": m,
        #     "pre_gflops": pre_gflops,
        #     "pre_time": pre_time,
        #     "adj_gflops": adj_gflops,
        #     "adj_time": adj_time,
        #     "pre_acc": pre_acc,
        #     "pre_ukr": pre_ukr,
        #     "pre_packa": pre_packa,
        #     "pre_packb": pre_packb,
        #     "adj_acc": adj_acc,
        #     "adj_ukr": adj_ukr,
        #     "adj_packa": adj_packa,
        #     "adj_packb": adj_packb,
        # }
        

        # print(output_txt[13], "\n", output_txt[17])

    print(outputs)

if __name__ == "__main__":
    main()
