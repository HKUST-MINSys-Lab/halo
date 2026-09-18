# MM-Fit scenario source

MM-Fit is retained as a scenario-only multi-device benchmark. Its four synchronized IMU streams are
left/right smartwatch, pocket phone and earbud. Do not make cross-subject claims from arbitrary
workout identifiers: the public release maps 21 workouts to ten people incompletely.

`eval_protocol.json` records the publication's participant-disjoint query workouts and reference
workouts. Scenario construction must use that file for cross-device, device-set and gym-domain rows.

Reference: D. Stromback, S. Huang and V. Radu, *MM-Fit: Multimodal Deep Learning for Automatic
Exercise Logging across Sensing Devices*, IMWUT 2020. https://vradu.uk/UbiComp2021.pdf
