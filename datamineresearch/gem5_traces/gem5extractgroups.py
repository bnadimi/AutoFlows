def extract_groups_from_msg_file(file_path):
    group_indices = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue  # Ignore comments
            elif line:
                parts = [part.strip() for part in line.split(':')]
                # print(parts)
                if len(parts) == 4:
                    index, src, dest, type_ = parts
                    key = tuple(sorted((src, dest)))  # Sorting to consider src, dest and dest, src as the same
                    if key not in group_indices:
                        group_indices[key] = []
                    group_indices[key].append(int(index))

    groups = set()  # Avoid duplicates
    for key, indices in group_indices.items():
        src, dest = key
        group = (src, dest, tuple(sorted(indices)))  # Include the indices
        groups.add(group)

    return sorted(groups)  # Return sorted groups


file_path = "gem5_traces\defSnoop-RubelPrintFormat.msg"
groups = extract_groups_from_msg_file(file_path)
for g in groups:
    print(g)

# file_path = "gem5_traces\defThreads-RubelPrintFormat.msg"
# groups = extract_groups_from_msg_file(file_path)
# for g in groups:
#     print(g)