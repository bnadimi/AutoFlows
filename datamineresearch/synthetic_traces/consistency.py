import os

def read_txt_file(file_path):
    indices = {}
    with open(file_path, 'r') as file:
        for line in file:
            index_pair, _ = line.strip().split(' : ')
            indices[tuple(map(int, index_pair.split()))] = indices.get(tuple(map(int, index_pair.split())), 0) + 1
    return indices

def compare_indices(txt_files):
    all_indices = {}
    for file in txt_files:
        indices = read_txt_file(file)
        for index_pair, _ in indices.items():
            all_indices[index_pair] = all_indices.get(index_pair, 0) + 1
    
    sorted_indices = sorted(all_indices.items(), key=lambda x: x[0])
    
    print("Binary Patterns:")
    for index_pair, count in sorted_indices:
        print(f"{index_pair} : {count}")
    


if __name__ == "__main__":
    folder_path = "synthetic_traces/traces/binary patterns" 
    txt_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]
    compare_indices(txt_files)



