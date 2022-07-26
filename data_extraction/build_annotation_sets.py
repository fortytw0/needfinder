import collections
import csv
import random

Triplet = collections.namedtuple('Triplet',
  'index query adj_0 adj_1')

FIELDNAMES = "original_index presentation_index annotator is_flipped query adj_0 adj_1".split()

AnnotationExample = collections.namedtuple('AnnotationExample', FIELDNAMES)

def make_two_examples(triplet, flip_which):
  examples = [{
    'original_index': triplet.index,
    'query': triplet.query,
    'annotator': i,
    'is_flipped': i == flip_which,
  } for i in range(2)]

  ordered_adj_quotes = [
  [triplet.adj_0, triplet.adj_1],
  [triplet.adj_1, triplet.adj_0]]

  for i in range(2):
    if flip_which == 1:
      examples[i]['adj_quotes'] = ordered_adj_quotes[i]
    else:
      examples[i]['adj_quotes'] = ordered_adj_quotes[1-i]

  return examples


def main():
  adjacency_map = collections.defaultdict(list)
  with open('results-adjacency_vs_sim.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      if row['relation'] == 'adjacent':
        adjacency_map[row['query']].append(row['target'])


  sorted_queries = list(sorted(adjacency_map.keys()))
  data_triplets = []

  for query_i, query in enumerate(sorted_queries):
    maybe_adjacent_quotes = adjacency_map[query]
    if len(maybe_adjacent_quotes) == 1:
      continue
    adj_0, adj_1 = list(sorted(maybe_adjacent_quotes))
    data_triplets.append(Triplet(query_i, query, adj_0, adj_1))

  random.seed(33)

  examples = []

  for triplet in data_triplets:
    flip_which = random.choice([0, 1])
    examples += make_two_examples(triplet, flip_which)

  random.shuffle(examples)

  annotation_examples = []
  example_counter = [0,0]

  for i, example in enumerate(examples):
    annotation_examples.append(
      AnnotationExample(example['original_index'],
      example_counter[example['annotator']],
      example['annotator'],
      example['is_flipped'],
      example['query'],
      example['adj_quotes'][0],
      example['adj_quotes'][1]))
    example_counter[example['annotator']] += 1


  with open('annotation_examples.tsv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter='\t')
    writer.writeheader()
    for example in sorted(annotation_examples, key=lambda x:x.annotator):
      writer.writerow(example._asdict())





if __name__ == "__main__":
  main()

