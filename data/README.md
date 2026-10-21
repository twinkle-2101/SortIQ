# Filename dataset

`file_dataset.csv` contains 1,200 **synthetic** filename examples: 200 for each
of the six project categories. The records are generated reproducibly by
`generate_dataset.py`, using varied subject-specific filename stems, common
revision/date suffixes, and multiple file extensions.

The CSV contract is exactly `file_name`, `extension`, and `category`.
`file_name` contains a filename only, not a path. Extensions are lowercase and
include the leading dot. No real people's files or private filenames are used.

This is a constructed development dataset, not a sample of real user downloads.
Its size and generated naming patterns do not establish real-world accuracy.
Evaluation results should be reported with that limitation, and the dataset
should be replaced or supplemented with appropriately sourced, consented
examples before making claims about real-world performance.

Regenerate from the repository root with:

```powershell
python data\generate_dataset.py
```