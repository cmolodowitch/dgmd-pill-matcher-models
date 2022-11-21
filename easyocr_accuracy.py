import os

import test_utils as utils
import easyocr_predict as predict


def run_test() -> None:
    # Parent directory where everything else is located
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"

    # Set gpu False to use CPU only
    # Set model_path with directory where model files should be downloaded
    ocr = predict.generate_ocr()

    text_file_name = "pill_labels_challenge.txt"
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_c3pi_to_dataframe(text_file, parent_folder + r"\images\C3PI full data")
    print(labels.head())
    print(labels.info())

    sample_display, labels = utils.sample_images(labels, sample=False)
    # sample_display = "4_select"
    # labels = labels[labels["image_file"].isin(["ROUG9WZMXRIHLLLGNG154WL4VHLN2K.JPG",
    #                                            "RP6UURAWU3ZRAX-W1J!0RYAH85424_.JPG",
    #                                            "RPAP_T-K!E_HCJDIKLUXJ_Q3U0UT5H.JPG",
    #                                            "RPFWRMA0ZQZFG92QSNFDCO-GOAF6F!.JPG"])]
    # print(labels.head())

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"easyocr_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(output_file)

    utils.run_test(ocr, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    run_test()
