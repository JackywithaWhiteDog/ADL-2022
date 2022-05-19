import sys

if __name__ == '__main__':
    source_file, input_file, output_file = sys.argv[1:]
    sources = []
    with open(source_file, 'r') as f:
        for line in f:
            sources.append(line[line.find(",")+1:].strip())
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(line.strip())
    results = [f'{s} {d}' for s, d in zip(sources, data)]
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
