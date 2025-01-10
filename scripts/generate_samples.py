import os
import random
import time
from multiprocessing import Pool, cpu_count
from textattack.transformations import (
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapQWERTY,
)
from textattack.shared import AttackedText
input_dir = '/tmp/project_996/cs_code/'
transformations = [
    WordSwapEmbedding(),
    WordSwapHomoglyphSwap(),
    WordSwapNeighboringCharacterSwap(),
    WordSwapRandomCharacterDeletion(),
    WordSwapQWERTY(),
]
def generate_adversarial_text(text):
    try:
        attacked_text = AttackedText(text)
        transformation = random.choice(transformations)
        transformed_texts = transformation(attacked_text)
        if transformed_texts:
            return transformed_texts[0].text
    except Exception as e:
        print(f"Error in transformation: {e}")
    return text
def process_file(args):
    input_file, index, total_files = args
    error_log_path = os.path.join(input_dir, "error_log.txt")
    try:
        temp_file = f"{input_file}.tmp"
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        modified_lines = [generate_adversarial_text(line) for line in lines]
        with open(temp_file, 'w', encoding='utf-8') as temp_f:
            temp_f.writelines(modified_lines)
        os.replace(temp_file, input_file)
        return input_file, None
    except Exception as e:
        print(f"Unexpected error while processing {input_file}: {e}")
        with open(error_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Error in {input_file}: {e}\n")
        return input_file, str(e)

all_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.endswith('.code')]
total_files = len(all_files)
selected_files = random.sample(all_files, int(0.1 * total_files))
if __name__ == '__main__':
    start_time = time.time()

    num_cores = cpu_count() // 2

    tasks = [(file, idx + 1, len(selected_files)) for idx, file in enumerate(selected_files)]

    with Pool(num_cores) as pool:
        results = pool.map(process_file, tasks)
    error_count = 0
    error_log_path = os.path.join(input_dir, "error_log.txt")
    with open(error_log_path, "a", encoding="utf-8") as log_file:
        for file, error in results:
            if error:
                error_count += 1
                log_file.write(f"Error in {file}: {error}\n")
    elapsed_time = time.time() - start_time

