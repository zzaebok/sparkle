import os
import bz2
import gzip
from tqdm import tqdm
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

def read_file_in_chunks(input_path, chunk_size):
    chunk = []

    with bz2.BZ2File(input_path) as file:
        for i, line in tqdm(enumerate(file)):
            line = line.decode().strip()

            if line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]

            
            component = json.loads(line)

            result = dict()
            result["id"] = component["id"]
            result["type"] = component.get("type", "None")
            if "labels" in component and "en" in component["labels"]:
                result["labels"] = {"en": component["labels"]["en"]}
            else:
                result["labels"] = dict()
            if "claims" in component:
                result["claims"] = component["claims"]
            
            chunk.append(result)
            if len(chunk) == chunk_size:
                yield json.dumps(chunk)
                chunk = []
        if chunk:
            yield json.dumps(chunk)


def write_chunk(chunk, output_path):
    with gzip.open(output_path, "wt") as f_out:
        f_out.write(chunk)

def main():
    wikidata_path = f"{os.getenv('DATA_PATH')}/original/wikidata/wikidata20171227.json.bz2"
    part_dir=f"{os.getenv('DATA_PATH')}/original/wikidata/parts"
    chunk_size = 500_000

    with ThreadPoolExecutor(int(os.cpu_count() * 0.8)) as executor:
        futures = []
        for index, chunk in enumerate(
            read_file_in_chunks(wikidata_path, chunk_size)
        ):
            output_path = os.path.join(part_dir, f'part_{index}.json.gz')

            future = executor.submit(write_chunk, chunk, output_path)
            futures.append(future)

        # print number of completed jobs using tqdm
        for _ in tqdm(concurrent.futures.as_completed(futures)):
            pass

if __name__ == '__main__':
    main()
