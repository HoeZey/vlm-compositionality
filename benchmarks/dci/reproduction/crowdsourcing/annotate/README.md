# Long Captions Annotations
This task is used to create long captions, which are highly aligned image<->text pairings containing a target average of 750 words per image.

This portion of the project additionally requires the [segment-anything](https://github.com/facebookresearch/segment-anything) library to do mask prediction. It also requires opencv (`pip install opencv-python`)

## Data Format
The final data format (as exported by `export_data.py`) has the following structure:
```yaml
{
    "image": "relative-image-path.jpg",
    "short_caption": "A standard short-form caption for the image",
    "mask_data": {
      "mask_key": {
        "idx": "mask_key", # Self-reference into mapping
        "outer_mask": "iVBORw0KGgoAAAANSUhE.....", # base64 encoding of the binary mask for this segment
        "mask_quality": 0, # one of 0, 1, or 2 for "ok", "low-quality/uninteresting", or "bad" respectively
        "label": "A short label for the given mask", # omitted if "bad" quality
        "caption": "A long descriptive caption for this given mask", # only for "ok" masks
        "parent": "other_mask_key", # either the parent mask id in the tree, or -1 if parent is the base image
        "requirements": ["list", "of", "children", "masks"] # mask IDs for children masks
        "bounds": [[0, 0], [500, 500]] # TopLeft & BottomRight coords of mask bounds
        "area": 123, # mask size in pixels 
      },
      # ...
    },
    "mask_keys": ["list", "of", "mask_keys", "into", "mask_data"],
    "extra_caption": "Additional long form caption that may catch additional information about layout or from from missing masks",
}
```

## Task Setup
### Preprocessing
Images to be annotated should first be run through the `preprocessing/preprocess_assets_segev.py` script, which applies the SAM mask generator on the provided images, then uses additional heuristics to construct a tree of submasks for the full image. This will be used for the annotation step.

### Core task
Workers are given the main photo and asked to do three primary steps:
1. Provide a short caption for the image
2. When provided a segment (generated by SAM) of the image, provide a label and descriptive caption for the contents shown
3. After seeing all of the segments, provide one last caption to cover any descriptive details that have been missed.

## Execution
For testing, you can run this script with:
```bash
python run_task.py conf=test
```

For a live run, this task is expected to be run alongside the `qualify` version, to pre-qualify a batch of workers and remove spammers and low-quality bots upfront. Assuming you already have that task run and data configured, you can launch a set of the task to allow workers to do 3 test tasks which can be evaluated for quality before moving to the full task batch. The following script can be used assuming you have set up a mephisto profile for your live launch details:
```bash
python run_task.py conf=pilot +profile=live
```

After reviewing enough data, you can launch a larger batch using the following:
```bash
python run_task.py conf=live +profile=live
```

## Review
You can review using Mephisto v1.2's built-in review, introduced [in this PR](https://github.com/facebookresearch/Mephisto/pull/1058).

## Viewing
You can also directly view (exported) data from the task using the `explorer` folder, which contains a `run_server.py` script and a `view` folder. After running `pip install flask pillow`, you should be able to execute `python run_server.py` from the root directory.