# MobiAct v2 prospective evaluation source

This directory accepts only the official MobiAct v2 annotated CSV release. MobiFall text files are
not interchangeable with MobiAct and the converter rejects them.

1. Obtain access through the [official project page](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/).
2. Run `python3 -m data.datasets.mobiact.setup --archive /absolute/path/to/archive.zip`.
3. The setup hashes the archive, validates every CSV, freezes a subject-disjoint reference/query
   split, builds native 4/8/16-second grids, derives duration-qualified candidate panels, and
   refreshes the quality screens required by evaluation.

The source is a prospective evaluation benchmark. It must be evaluated with
`training.support_classifier.sealed_eval --scope prospective`; it is not included in historical
six-dataset aggregate means.

Reference: G. Vavoulas et al., *The MobiAct Dataset: Recognition of Activities of Daily Living using
Smartphones*, ICT4AWE 2016. https://www.scitepress.org/PublishedPapers/2016/57924/57924.pdf
