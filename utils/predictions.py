from pathlib import Path

import pandas as pd


def save_predictions(
    predictions,
    filenames,
    output_path: Path,
    use_filenames: bool = True,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_filenames:
        df = pd.DataFrame({"filename": filenames, "prediction": predictions})
    else:
        df = pd.DataFrame({"prediction": predictions})

    df.to_csv(output_path, index=False)
