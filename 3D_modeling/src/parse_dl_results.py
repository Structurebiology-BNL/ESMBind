import pickle
import os
import numpy as np
import argparse

# Dictionary of optimal thresholds for each ion
# the optimal threshold is calculated by the maximum F1 score
OPTIMAL_THRESHOLDS = {
    "MG": 0.562,
    "FE": 0.778,
    "CU": 0.628,
    "CO": 0.554,
    "CA": 0.614,
    "MN": 0.770,
    "ZN": 0.760,
}


def parse_predictions(predictions_file, ion, lower_factor=1.0):
    # Read the prediction pickle file
    with open(predictions_file, "rb") as f:
        predictions = pickle.load(f)

    # Get the list of IDs
    ID_list = list(predictions[ion].keys())

    parsed_result = {}

    # Calculate the threshold
    optimal_threshold = OPTIMAL_THRESHOLDS[ion]
    chosen_threshold = optimal_threshold * lower_factor

    # Iterate over IDs to calculate the statistics
    for id in ID_list:
        # Get the prediction results above the chosen threshold
        prediction_indices = np.where(predictions[ion][id] > chosen_threshold)[0] + 1

        # If there are predictions for the ion, add them to parsed_result
        if prediction_indices.size > 0:
            parsed_result[id] = prediction_indices.tolist()

    return chosen_threshold, parsed_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the deep learning predictions for real-world application."
    )
    parser.add_argument(
        "ion",
        type=str,
        choices=["CA", "ZN", "MG", "MN", "FE", "CU", "CO"],
        help="Type of ion to analyze.",
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to the predictions pickle file.",
    )
    parser.add_argument(
        "--lower_factor",
        type=float,
        default=0.5,
        help="Lower factor to adjust the threshold. Set to 1.0 to use the original threshold.",
    )
    args = parser.parse_args()

    ion = args.ion.upper()
    predictions_file_path = args.predictions_file
    # we recommend to use the lower factor of 0.5 to decrease false negatives
    lower_factor = args.lower_factor

    # Parse predictions
    threshold, parsed_result = parse_predictions(
        predictions_file_path, ion, lower_factor=lower_factor
    )

    print(f"# of parsed sequences is {len(parsed_result)} for {ion}")
    print(f"Used threshold: {threshold:.2f}")

    # Generate output file name based on input file name
    input_file_name = os.path.basename(predictions_file_path)
    output_file_name = f"parsed_result_{input_file_name.split('.')[0]}_{ion}_lower_factor_{lower_factor:.2f}.pkl"
    output_file_path = os.path.join(
        os.path.dirname(predictions_file_path), output_file_name
    )

    # Save parsed_result to a file
    with open(output_file_path, "wb") as f:
        pickle.dump(parsed_result, f)

    print(f"Parsed results saved to: {output_file_path}")
