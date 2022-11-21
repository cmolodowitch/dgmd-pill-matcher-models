import os

import test_utils as utils
import keras_predict as predict


def run_test() -> None:
    # Parent directory where everything else is located
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    # Set environment variable KERAS_OCR_CACHE_DIR to specify where the weights get downloaded
    pipeline = predict.initialize()

    text_file_name = "pill_labels_full_clear.txt"
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_c3pi_to_dataframe(text_file, parent_folder + r"\images\C3PI full data")
    print(labels.head())
    print(labels.info())

    sample_display, labels = utils.sample_images(labels, sample=True, head=True, n=1)

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"keras_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(output_file)

    utils.run_test(pipeline, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    run_test()
