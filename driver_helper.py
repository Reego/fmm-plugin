import os
import subprocess

def get_line_0(line):

    line = line.decode('utf-8')
    start = line.index("GFLOPS:") + 8
    end = line.index("Total")
    next_start = end + 6

    gflops = line[start:end].strip()
    total = line[next_start:].strip()

    return gflops, total

def get_line_1(line):

    line = line.decode('utf-8')

    start_acc = line.index("ACC") + 3
    end_acc = line.index("UKR_TOTAL")

    acc = line[start_acc:end_acc].strip()

    start_ukr = line.index("UKR_TOTAL") + 9
    end_ukr = line.index("PACKB")

    ukr = line[start_ukr:end_ukr].strip()

    start_packb = line.index("PACKB") + 5
    end_packb = line.index("PACKA")

    pack_b = line[start_packb:end_packb].strip()

    start_packa = line.index("PACKA") + 5

    pack_a = line[start_packa:].strip()

    return acc, ukr, pack_b, pack_a
def main():

    outputs = dict()
    
    for m in range(240, 3500, 240):

        data = subprocess.run(f"./driver.x {m} {m} {m} -f 1 222.txt", capture_output=True, shell=True)

        output = data.stdout.splitlines()

        outputs[m] = {

        }
        
        pre_line_0 = output[12]
        pre_line_1 = output[13]

        adj_line_0 = output[17]
        adj_line_1 = output[18]
        
        pre_gflops, pre_time = get_line_0(pre_line_0)
        pre_acc, pre_ukr, pre_packa, pre_packb = get_line_1(pre_line_1)

        adj_gflops, adj_time = get_line_0(adj_line_0)
        adj_acc, adj_ukr, adj_packa, adj_packb = get_line_1(adj_line_1)

        outputs[m] = {
            "m": m,
            "pre_gflops": pre_gflops,
            "pre_time": pre_time,
            "adj_gflops": adj_gflops,
            "adj_time": adj_time,
            "pre_acc": pre_acc,
            "pre_ukr": pre_ukr,
            "pre_packa": pre_packa,
            "pre_packb": pre_packb,
            "adj_acc": adj_acc,
            "adj_ukr": adj_ukr,
            "adj_packa": adj_packa,
            "adj_packb": adj_packb,
        }
        

        # print(output_txt[13], "\n", output_txt[17])

    print(outputs)

if __name__ == "__main__":
    main()
