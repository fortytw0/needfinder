import csv

LABEL_MAPPER = {
  "Both equally similar": "both",
	"Neither is similar": "neither",
	"Problem": "problem"
}

def normalize_row(row):
  left, right = row['adj_0'], row['adj_1']
  more_similar = None
  if 'Left' in row['Judgment']:
    more_similar = left
  elif 'Right' in row['Judgment']:
    more_similar = right
  if row['is_flipped'] == "TRUE":
    q0, q1 = right, left
  else:
    q0, q1 = left, right

  assert [q0, q1] == sorted([q0, q1])

  if more_similar == q0:
    label = "q0_more_similar"
  elif more_similar == q1:
    label = "q1_more_similar"
  else:
    assert more_similar is None
    label = LABEL_MAPPER[row['Judgment']]
    assert not label == "problem"

  return q0, q1, label

def get_result_list(annotator):
  filename = f'{annotator}_labels.tsv'
  with open(filename, 'r') as f:
    r = csv.DictReader(f, delimiter='\t')
    return sorted([row for row in r if row['original_index']], key=lambda x:x['original_index'])

def main():

  result_lists = [
    get_result_list(annotator)
    for annotator in ["abe", "neha"]
  ]

  result_fieldnames = "original_index query q_0 q_1 label_0 label_1".split()

  with open('result.tsv', 'w') as f:
    writer = csv.DictWriter(f, result_fieldnames, delimiter="\t")
    writer.writeheader()
    for row_1, row_2 in zip(*result_lists):
      assert row_1['original_index'] == row_2['original_index']
      assert not row_1['is_flipped'] == row_2['is_flipped']
      assert not row_1['annotator'] == row_2['annotator']
      assert row_1['query'] == row_2['query']
      assert set([row_1['annotator'], row_2['annotator']]) == set(["0", "1"])
      q_0_0, q_1_0, label_0 = normalize_row(row_1)
      q_0_1, q_1_1, label_1 = normalize_row(row_2)
      assert q_0_0 == q_0_1 and q_1_0 == q_1_1
      writer.writerow(
      {
        "original_index": row_1["original_index"],
        "query": row_1["query"],
        "q_0": q_0_0,
        "q_1": q_1_0,
        "label_0": label_0,
        "label_1": label_1,
      }
      )


if __name__ == "__main__":
  main()

