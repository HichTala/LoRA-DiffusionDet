import datasets

CATEGORY_NAMES = []

FEATURES = datasets.Features(
    {
        "image_id": datasets.Value("int64"),
        "image": datasets.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "objects": datasets.Sequence({
            "bbox_id": datasets.Value("int64"),
            "category": datasets.ClassLabel(names=CATEGORY_NAMES),
            "bbox": datasets.Sequence(datasets.Value("int64"), 4),
            "area": datasets.Value("int64"),
        })
    }
)

ANNOTATIONS_PATH = ""
IMAGES_PATH = ""