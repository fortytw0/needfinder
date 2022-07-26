import collections
import csv
import glob
import xml.etree.ElementTree as ET

PREFIX = "{http://www.tei-c.org/ns/1.0}"
AUTHOR_ID = f'{PREFIX}author'

FIELDNAMES = "first_name first_email last_name last_email title".split()
PaperInfo = collections.namedtuple("PaperInfo", FIELDNAMES)

def first_child(parent, child_part):
  return parent.findall(f'{PREFIX}{child_part}')[0]

def get_author_info(author):
  try:
    name_pieces = [t.text for t in first_child(author, "persName")]
  except IndexError:
    name_pieces = ["None"]
  try:
    email = first_child(author, "email").text
  except IndexError:
    email = None
  return " ".join(name_pieces), email

def get_title(root):
  return first_child(first_child(first_child(first_child(root, "teiHeader"), "fileDesc"), "titleStmt"),
  "title").text

def get_docs(filename):

  root = ET.parse(filename).getroot()
  authors = first_child(first_child(first_child(first_child(first_child(root, "teiHeader"),
  "fileDesc"), "sourceDesc"), "biblStruct"), "analytic").findall(AUTHOR_ID)
  first_author_name, first_author_email = get_author_info(authors[0])
  last_author_name, last_author_email = get_author_info(authors[-1])
  return PaperInfo(
    first_author_name, first_author_email, last_author_name, last_author_email,
    get_title(root)
  )

with open("authors.tsv", 'w') as f:
  writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter='\t')
  writer.writeheader()
  for filename in glob.glob("xmls/*.xml"):
    writer.writerow(get_docs(filename)._asdict())



