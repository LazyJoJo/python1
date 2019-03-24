import pandas as pd
import xml.etree.ElementTree as ef

def xml_test():
    # tree = ef.parse('~/Desktop/test/capacity_cap.xml')
    # print(tree)
    print("")


def main():


    reader = pd.read_csv('~/Desktop/ans.csv',iterator=True,chunksize=10)
    print(reader)
    # chunkSize = 100
    # chunks = []
    # while True:
    #     try:
    #         chunk = reader.get_chunk(chunkSize)
    #         chunks.append(chunk)
    #     except StopIteration:
    #         print('Iteration is stopped')
    #         break

    df = pd.concat(reader,ignore_index=True)

    print(df)

if __name__ == '__main__':
    xml_test()