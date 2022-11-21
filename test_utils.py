import os
import csv
import re
from typing import List, Tuple
import time
import Levenshtein
import pandas as pd


def read_labels_file(labels_filepath: str, parent_folder: str) -> List[Tuple[str, str]]:
    """Read a labels file and return (filepath, label) tuples.

    Labels file assumed to be in the format <Image file name>, "<imprint text>"
    Imprint text expected to list all imprint items separated by semicolons

    Args:
        labels_filepath: Path to labels file
        parent_folder: Path to parent folder containing the image files
    """
    with open(labels_filepath, encoding="utf-8-sig") as f:
        labels_raw = [l.strip().split(",") for l in f.readlines()]
        labels = [
            (
                os.path.join(parent_folder, segments[0].strip()),
                ",".join(segments[1:]).strip()[1:-1],
            )
            for segments in labels_raw
        ]
    return labels


def read_c3pi_labels_file(labels_filepath, parent_folder) -> List[Tuple[str, str, str]]:
    """
    Read a labels file and return (filepath, label) tuples.

    Labels file assumed to be in the format <C3PI directory>, <Image file name>, <imprint rating>, "<imprint text>"
    Imprint text expected to list all imprint items separated by semicolons

    :param labels_filepath: Path to labels file
    :param parent_folder: Path to parent folder containing the C3PI data in the original C3PI directory structure
    :return: List of tuples, with the full path to the image file as first element, the imprint rating as second
             element, and the imprint text as third element
    """
    with open(labels_filepath, encoding="utf-8-sig") as f:
        labels_raw = [l.strip().split(",") for l in f.readlines()]
        labels = [
            (
                os.path.join(parent_folder, segments[0], "images", segments[1].strip()),
                segments[2],
                ",".join(segments[3:]).strip()[1:-1],
            )
            for segments in labels_raw
        ]
    return labels


def read_c3pi_to_dataframe(labels_filepath: str, parent_image_dir: str) -> pd.DataFrame:
    """
    Read a labels file and return a pandas DataFrame with a new "file_path" column containing the full path to the
    image file in that entry.

    Labels file assumed to be in the format <C3PI directory>, <Image file name>, <imprint rating>, "<imprint text>"
    Imprint text expected to list all imprint items separated by semicolons

    :param labels_filepath: Path to labels file
    :param parent_image_dir: Path to parent folder containing the C3PI data in the original C3PI directory structure
    :return: Pandas DataFrame containing the elements from the original CSV file, plus a new "file_path" column
             containing the full path to the image file in each entry
    """
    labels = pd.read_csv(labels_filepath,
                         names=["image_dir", "image_file", "imprint_rating", "imprint"],
                         quotechar='"',
                         dtype="string")

    labels["file_path"] = \
        labels.apply(lambda row: os.path.join(parent_image_dir, row["image_dir"], "images", row["image_file"]),
                     axis=1)
    labels = labels.astype("string")
    labels = labels.fillna("Empty")
    return labels


def sample_images(labels: pd.DataFrame, sample=True, head=True, n=50) -> (str, pd.DataFrame):
    sample_display = ""
    if sample:
        if head:
            labels = labels.head(n=n)
            sample_display = f"_{n}"
        else:
            labels = labels.sample(n=n)
            sample_display = f"_{n}_random"
    return sample_display, labels


def create_accuracy_tracking_dict():
    return {
        "Clear": {
            "total": 0.0,
            "count": 0
        },
        "Partial": {
            "total": 0.0,
            "count": 0
        },
        "Empty": {
            "total": 0.0,
            "count": 0
        }
    }


def find_strict_match(prediction: str, imprint_sections: List[str]) -> (int, float):
    """
    Find the imprint section that is identical to the predicted text.

    :param prediction: Predicted text, should not be null/empty
    :param imprint_sections: List of all the imprint sections
    :return: Tuple containing the index of the imprint section that is identical to the prediction as the first element
             and 1.0 (the accuracy factor) as the second element, or (-1, 0.0) if none of the imprint sections are
             identical to the prediction
    """
    for i, imprint in enumerate(imprint_sections):
        if prediction == imprint:
            return i, 1.0
    return -1, 0.0


def generate_regex_chars(prediction: str) -> List[str]:
    regex_chars = []
    for char in prediction:
        # o and 0 are frequently confused
        if char in {"o", "0"}:
            regex_chars.append("[o0]")
        # e, 3, 8, s, and 5 seem to be often interchanged
        elif char in {"e", "3", "8", "5", "s"}:
            regex_chars.append("[e385s]")
        # 1, i, and l are frequently confused
        elif char in {"1", "l", "i"}:
            regex_chars.append("[1il]")
        elif char in {"4", "a"}:
            regex_chars.append("[4a]")
        else:
            regex_chars.append(char)

    return regex_chars


def match_by_regex(regex_chars: List[str], imprint_sections: List[str], n_subs: int) -> (int, float):
    # Generate regular expressions based on the number of allowed "substitute" characters
    patterns = []
    # Add "$" to the end of each regex string to ensure we're only matching the whole string, not just the beginning
    if n_subs == 0:
        patterns.append(re.compile("".join(regex_chars) + "$"))
    elif n_subs == 1:
        for i in range(len(regex_chars)):
            regex = ""
            for j, regex_char in enumerate(regex_chars):
                if j == i:
                    regex = regex + "."
                else:
                    regex = regex + regex_char
            patterns.append(re.compile(regex + "$"))

    # Reduce accuracy by 3/4 if using a regex
    accuracy_factor = 0.75 * (1 - (n_subs / len(regex_chars)))

    for i, imprint in enumerate(imprint_sections):
        for pattern in patterns:
            if pattern.match(imprint):
                return i, accuracy_factor
    return -1, 0.0


def match_substring(prediction, imprint_sections: List[str]) -> (int, float):
    for i, imprint in enumerate(imprint_sections):
        if prediction in imprint:
            # Reduce accuracy based on the actual number of matched characters, times further factor of 0.75
            return i, 0.75 * len(prediction) / len(imprint)
    return -1, 0.0


def find_loose_match(prediction: str, imprint_sections: List[str]) -> (int, float):
    index, factor = find_strict_match(prediction, imprint_sections)
    if index > -1:
        return index, factor
    else:
        regex_chars = generate_regex_chars(prediction)
        index, factor = match_by_regex(regex_chars, imprint_sections, 0)
        if index > -1:
            return index, factor
        else:
            index, factor = match_by_regex(regex_chars, imprint_sections, 1)
            if index > -1:
                return index, factor
            else:
                return match_substring(prediction, imprint_sections)


def find_distance(prediction: str, imprint_sections: List[str]) -> (int, float):
    # Try using the Levenshtein distance to match, with a cutoff of 0.5
    # No need to first check exact match, since that will have a distance of 1.0
    # and end up being the best distance
    index = -1
    best_distance = 0.0

    for i, section in enumerate(imprint_sections):
        # The ratio function returns a normalized similarity in range [0, 1], and is 1 - normalized distance
        # So perfect match = 1.0, failed match = 0.0
        distance = Levenshtein.ratio(prediction, section, score_cutoff=0.5)
        if distance > best_distance:
            index = i
            best_distance = distance

    return index, best_distance


def calc_accuracy(predictions: List[str], imprint: str, match_imprint) -> float:
    # Convert the imprint to lower case to avoid case issues and break it into separate portions using the
    # semicolon delimiter
    imprint_sections = imprint.lower().split(";")
    # Convert predictions into lower case as well to make matching easier
    predictions = [prediction.lower() for prediction in predictions]

    num_sections_matched = 0.0
    sections = imprint_sections.copy()
    if len(predictions) > 0:
        for prediction in predictions:
            if len(prediction.strip()):
                matching_index, factor = match_imprint(prediction, sections)
                # If a match is found, remove that element from the list of imprint sections in case the prediction has
                # duplicates but the imprints don't - that way only the first one will be counted as a successful match
                if matching_index > -1:
                    print(f"Prediction: {prediction}, match: {sections[matching_index]}, factor: {factor}")
                    num_sections_matched = num_sections_matched + factor
                    del sections[matching_index]

    return num_sections_matched / len(imprint_sections)


def test_image(ocr, image_file: str, imprint_rating: str, imprint: str, output_file: str,
               do_ocr, match_imprint) -> float:
    print(image_file)
    start = time.time()
    all_predictions = do_ocr(ocr, image_file, rotate=True)
    stop = time.time()

    prediction_outputs = []
    highest_accuracy = 0.0
    for prediction_group in all_predictions:
        accuracy = calc_accuracy(prediction_group, imprint, match_imprint)
        prediction_outputs.append(";".join(prediction_group))
        prediction_outputs.append(accuracy)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy

    with open(output_file, "a", newline="") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([image_file, imprint_rating, stop - start, imprint, highest_accuracy] + prediction_outputs)
        # outfile.write(
        #     f"{image_file},{imprint_rating},{stop-start},\"{imprint}\",\"{highest_accuracy}\",{','.join(prediction_strings)}\n")

    return highest_accuracy


def run_test(ocr, labels: pd.DataFrame, output_file: str, do_ocr, calc_match):
    total_accuracy = 0.0
    accuracy_tracking = create_accuracy_tracking_dict()

    for row in labels.itertuples():
        accuracy = test_image(ocr, row.file_path, row.imprint_rating, row.imprint, output_file,
                              do_ocr, calc_match)
        total_accuracy = total_accuracy + accuracy
        accuracy_sub = accuracy_tracking[row.imprint_rating]
        accuracy_sub["total"] = accuracy_sub["total"] + accuracy
        accuracy_sub["count"] = accuracy_sub["count"] + 1

    accuracy_output = f"Overall accuracy: {total_accuracy / len(labels)}"
    for rating in accuracy_tracking:
        rating_data = accuracy_tracking[rating]
        if rating_data["count"] > 0:
            rating_accuracy = rating_data["total"] / rating_data["count"]
        else:
            rating_accuracy = "No values"
        accuracy_output = accuracy_output + f"  {rating} accuracy: {rating_accuracy}"

    print(accuracy_output)
    with open(output_file, "a") as file:
        file.write(accuracy_output + "\n")
