from pathlib import Path
import yaml
import config
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
def load_data_yaml():
    """
    Load dataset/data.yaml.
    """
    if not config.DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nDataset YAML not found:\n"
            f"{config.DATA_YAML}"
        )
    with open(
        config.DATA_YAML,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(
            "data.yaml does not contain a valid YAML dictionary."
        )
    return data
def get_class_names():
    """
    Return class names from data.yaml.
    """
    data = load_data_yaml()
    names = data.get("names")
    if names is None:
        raise ValueError(
            "The data.yaml file must contain a 'names' field."
        )
    if isinstance(names, dict):
        names = [
            names[index]
            for index in sorted(
                names.keys(),
                key=lambda x: int(x),
            )
        ]
    if not isinstance(names, list):
        raise ValueError(
            "'names' in data.yaml must be a list or dictionary."
        )

    return names
def find_images(directory):
    """
    Recursively find supported images.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    images = []
    for path in directory.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)
    return sorted(images)
def validate_split(
    split_name,
    image_directory,
    label_directory,
):
    """
    Validate one dataset split.
    """

    image_directory = Path(image_directory)
    label_directory = Path(label_directory)

    if not image_directory.exists():
        raise FileNotFoundError(
            f"\n{split_name} image directory not found:\n"
            f"{image_directory}"
        )

    if not label_directory.exists():
        raise FileNotFoundError(
            f"\n{split_name} label directory not found:\n"
            f"{label_directory}"
        )

    images = find_images(
        image_directory
    )

    if not images:
        raise RuntimeError(
            f"\nNo images found in {split_name} split:\n"
            f"{image_directory}"
        )

    missing_labels = []
    invalid_labels = []

    class_names = get_class_names()
    number_of_classes = len(
        class_names
    )

    for image_path in images:

        relative_path = image_path.relative_to(
            image_directory
        )

        label_path = (
            label_directory
            / relative_path.with_suffix(".txt")
        )

        if not label_path.exists():

            missing_labels.append(
                str(relative_path)
            )

            continue

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    values = line.split()

                    if len(values) != 5:

                        invalid_labels.append(
                            (
                                str(label_path),
                                line_number,
                                "Expected 5 values.",
                            )
                        )

                        continue

                    try:

                        class_id = int(
                            values[0]
                        )

                        coordinates = [
                            float(value)
                            for value in values[1:]
                        ]

                    except ValueError:

                        invalid_labels.append(
                            (
                                str(label_path),
                                line_number,
                                "Non-numeric value found.",
                            )
                        )

                        continue

                    if not (
                        0 <= class_id < number_of_classes
                    ):

                        invalid_labels.append(
                            (
                                str(label_path),
                                line_number,
                                (
                                    f"Invalid class ID "
                                    f"{class_id}."
                                ),
                            )
                        )

                    for coordinate in coordinates:

                        if not (
                            0.0 <= coordinate <= 1.0
                        ):

                            invalid_labels.append(
                                (
                                    str(label_path),
                                    line_number,
                                    (
                                        "YOLO coordinates must "
                                        "be between 0 and 1."
                                    ),
                                )
                            )

        except OSError as error:

            invalid_labels.append(
                (
                    str(label_path),
                    0,
                    f"Could not read label file: {error}",
                )
            )

    if missing_labels:

        print(
            f"\nWARNING: {len(missing_labels)} "
            f"{split_name} images have no label file."
        )

        for path in missing_labels[:10]:

            print(
                f"  Missing: {path}"
            )

        if len(missing_labels) > 10:

            print(
                f"  ... and "
                f"{len(missing_labels) - 10} more."
            )

    if invalid_labels:

        print(
            f"\nWARNING: {len(invalid_labels)} "
            f"invalid label entries in {split_name}."
        )

        for item in invalid_labels[:10]:

            print(
                f"  {item[0]} "
                f"(line {item[1]}): "
                f"{item[2]}"
            )

        if len(invalid_labels) > 10:

            print(
                f"  ... and "
                f"{len(invalid_labels) - 10} more."
            )

    return {
        "split": split_name,
        "images": len(images),
        "missing_labels": len(missing_labels),
        "invalid_labels": len(invalid_labels),
    }
def validate_dataset():
    """
    Validate train, validation and independent test sets.
    """
    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    print(f"Dataset directory: {config.DATASET_DIR}")
    print(f"Data YAML:         {config.DATA_YAML}")
    class_names = get_class_names()
    print("\nClasses:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")
    print("\nValidating train split...")
    train_info = validate_split(
        "train",
        config.TRAIN_IMAGES_DIR,
        config.TRAIN_LABELS_DIR,
    )
    print("\nValidating validation split...")
    val_info = validate_split(
        "val",
        config.VAL_IMAGES_DIR,
        config.VAL_LABELS_DIR,
    )
    print("\nValidating test split...")
    test_info = validate_split(
        "test",
        config.TEST_IMAGES_DIR,
        config.TEST_LABELS_DIR,
    )
    print("\n" + "-" * 80)
    print("DATASET SUMMARY")
    print("-" * 80)
    print(
        f"Train images : {train_info['images']}"
    )
    print(
        f"Val images   : {val_info['images']}"
    )
    print(
        f"Test images  : {test_info['images']}"
    )
    print(
        f"Classes      : {len(class_names)}"
    )
    print("=" * 80)
    return {
        "names": class_names,
        "nc": len(class_names),
        "train": train_info,
        "val": val_info,
        "test": test_info,
    }
if __name__ == "__main__":
    config.create_directories()
    validate_dataset()