import sys
import json
import phrasemachine
from tqdm import tqdm

if __name__ == "__main__":

    fn = sys.argv[1]
    content_field = "body"


    data_phrases = []
    with open(fn, "r") as inf:
        with open(fn.replace(".jsonl", ".phrases.jsonl"), "w") as of:
            for post in tqdm(inf):
                post = json.loads(post)
                body = post[content_field]
                phrases = phrasemachine.get_phrases(body)["counts"]
                phrases = list(phrases.keys())
                post["phrases"] = phrases
                of.write(json.dumps(post) + "\n")