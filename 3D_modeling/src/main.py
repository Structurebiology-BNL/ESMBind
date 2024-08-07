import argparse
import pickle
import os
import contextlib
import tempfile
import traceback
from energy_minimization_openmm import perform_minimization
from add_ions import process_pdb_with_ions


def main(
    debug,
    use_gpu,
    pdb_dir,
    prediction_result,
    output_dir="./",
    gpu_index=0,
    ion="MN",
    restraint_force_constant=20920,
):
    with open(prediction_result, "rb") as f:
        predictions = pickle.load(f)
    temp_dir_path = "./" if debug else None
    predictions = predictions[ion]
    list_of_ids = list(predictions.keys())

    for id in list_of_ids:
        error_occurred = False
        print("processing", id)
        temp_context_manager = (
            tempfile.TemporaryDirectory()
            if not debug
            else contextlib.nullcontext(temp_dir_path)
        )

        with temp_context_manager as temp_dir:
            prmtop_file = os.path.join(temp_dir, "amber.prmtop")
            inpcrd_file = os.path.join(temp_dir, "amber.inpcrd")
            output_file = os.path.join(output_dir, f"{id}_with_{ion}_minimized.cif")
            if os.path.exists(output_file):
                print(
                    f"Output file {output_file} already exists. Skipping processing for {id}."
                )
                continue
            try:
                print(f"Processing PDB files for {id}...")
                ion_to_group = process_pdb_with_ions(
                    id,
                    "A",
                    predictions[id],
                    ion,
                    pdb_directory=pdb_dir,
                    temp_dir=temp_dir,
                    output_file=output_file,
                )
            except Exception as e:
                if debug:
                    print(f"Detailed error while processing PDB files for {id}:")
                    traceback.print_exc()
                else:
                    print(f"Error processing PDB files for {id}: {e}")
                error_occurred = True

            if not error_occurred:
                if not ion_to_group:
                    print(f"No ions were added for {id}. Skipping energy minimization.")
                    continue  # Skip to the next iteration of the loop

                for key in ion_to_group:
                    print(
                        f"Predicted binding residues for {ion}_{key}: {ion_to_group[key]}"
                    )

                print("\nStarting energy minimization...")
                try:
                    perform_minimization(
                        prmtop_file,
                        inpcrd_file,
                        output_file,
                        use_gpu=use_gpu,
                        gpu_index=str(gpu_index),
                        metal_ion_name=ion,
                        restraint_force_constant=restraint_force_constant,
                        verbose=True if debug else False,
                    )
                except Exception as e:
                    if debug:
                        print(f"Detailed error while performing minimization for {id}:")
                        traceback.print_exc()
                    else:
                        print(f"Error performing minimization for {id}: {e}")
                    error_occurred = True

            else:
                print("Use the DL predictions as the result.\n")

            if debug:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process debug and GPU usage flags.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU usage."
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="Specify the GPU ID to use. Default is 0.",
    )
    parser.add_argument(
        "--pdb-dir",
        dest="pdb_dir",
        default=os.getcwd(),
        help="Directory to search for PDB files. Default is the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=os.getcwd(),
        help="Directory to save the results. Default is the current directory.",
    )
    parser.add_argument(
        "--prediction_result",
        type=str,
        help="the prediction results from deep learning model",
    )
    parser.add_argument("--ion", type=str, help="Specify the ion type.", default="MN")
    parser.add_argument(
        "--restraint_force_constant",
        type=float,
        default=20920,
        help="Specify how much restraint to put on the metal ion and backbone. Default is 83680 kJ/(mol nm^2).",
    )
    args = parser.parse_args()

    main(
        debug=args.debug,
        use_gpu=args.use_gpu,
        gpu_index=args.gpu_index,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        prediction_result=args.prediction_result,
        ion=args.ion,
        restraint_force_constant=args.restraint_force_constant,
    )
