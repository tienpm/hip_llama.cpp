import argparse
import datetime
import logging
from pathlib import Path


def compare_files(file1_path, file2_path, log_path):
    """
    Compares two files line by line, logs differences to a file in a specified folder,
    and prints to console. (Using pathlib)
    """

    def read_paragraphs(file_path):
        with open(file_path, "r") as file:
            return file.read().strip().split("\n\n")

    # Configure logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f"comparison_{file1_path.stem}_{timestamp}.log"
    log_filepath = log_path / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
    )

    num_diff = 0
    num_prompt = 1

    paragraphs1 = read_paragraphs(file1_path)
    paragraphs2 = read_paragraphs(file2_path)

    if len(paragraphs1) > 0 and len(paragraphs2) > 0:
        if int(paragraphs1[0].splitlines()[0]) != int(paragraphs2[0].splitlines()[0]):
            raise Exception("Num prompts are not equal")
        else:
            num_prompt = int(paragraphs1[0].splitlines()[0])
            paragraphs1[0] = "\n".join(paragraphs1[0].splitlines()[1:])
            paragraphs2[0] = "\n".join(paragraphs2[0].splitlines()[1:])
    else:
        raise Exception("Empty file detected")

    # comparison logic
    line_no1 = 1
    line_no2 = 1
    for res_i in range(num_prompt):
        paragraph1 = paragraphs1[res_i].splitlines()
        paragraph2 = paragraphs2[res_i].splitlines()
        offset_line1 = len(paragraph1)
        offset_line2 = len(paragraph2)
        if len(paragraphs1) > len(paragraphs2):
            paragraph2 += ["" for _ in range(len(paragraph1) - len(paragraph2))]
        else:
            paragraph1 += ["" for _ in range(len(paragraph2) - len(paragraph1))]

        is_diff = False
        msg = ""
        for line_no, (line1, line2) in enumerate(zip(paragraph1, paragraph2)):
            if line1 != line2:
                is_diff = True
                # Split lines into words for finer comparison
                words1 = line1.split()
                words2 = line2.split()
                msg += f"{line_no1 + line_no + 1} - {line_no2 + line_no + 1}: "
                if len(words1) > len(words2):
                    words2 += ["" for _ in range(len(words1) - len(words2))]
                else:
                    words1 += ["" for _ in range(len(words2) - len(words1))]

                for i in range(len(words1)):
                    if words1[i] != words2[i]:
                        msg += f"({i}, {words1[i]}, {words2[i]}); "
                msg += "\n"

        if is_diff:
            num_diff += 1
            logging.info(f"Differences at respond {res_i + 1}:\n{msg}")

        line_no1 += offset_line1 + 1
        line_no2 += offset_line2 + 1

    test_cov = (num_prompt - num_diff) / num_prompt
    logging.info(f"NUM PROMPT: {num_prompt} - NUM DIFF: {num_diff}")
    logging.info(f"TEST COVERAGE: {test_cov}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Compare two files line by line and highlight differences."
    )
    parser.add_argument("file1", help="Path to the first file.")
    parser.add_argument("file2", help="Path to the second file.")
    parser.add_argument(
        "-l", "--log-folder", default="logs", help="Folder to store log files."
    )
    args = parser.parse_args()

    pwd = Path.cwd().resolve()
    log_folder = "logs"
    log_path = pwd / log_folder

    # Ensure log folder exists
    log_path.mkdir(parents=True, exist_ok=True)

    compare_files(Path(args.file1), Path(args.file2), log_path)
