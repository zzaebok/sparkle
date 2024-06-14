import os
import gzip
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


def read_file_in_chunks(input_path, chunk_size):
    chunk = []
    with gzip.open(input_path, "rt") as f:
        for line in tqdm(f):
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def write_chunk(chunk, output_path):
    with gzip.open(output_path, "wt") as f_out:
        f_out.writelines(chunk)

def main():
    freebase_path = f"{os.getenv('DATA_PATH')}/original/freebase/freebase-rdf-2015-08-09-00-01.gz"
    part_dir=f"{os.getenv('DATA_PATH')}/original/freebase/parts"
    chunk_size = 50_000_000

    with ThreadPoolExecutor(int(os.cpu_count() * 0.8)) as executor:
        futures = []
        for index, chunk in enumerate(
            read_file_in_chunks(freebase_path, chunk_size)
        ):
            output_path = os.path.join(part_dir, f'part_{index}.gz')

            future = executor.submit(write_chunk, chunk, output_path)
            futures.append(future)

        # print number of completed jobs using tqdm
        for _ in tqdm(concurrent.futures.as_completed(futures)):
            pass

if __name__ == '__main__':
    main()
