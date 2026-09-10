# nymeria_xsens

Converted output of [`../nymeria`](../nymeria), which fetches one Nymeria download and
splits it into two datasets because the device families record at different rates
(Xsens at 240 Hz, Aria at 200 Hz) and `metadata.json` carries a single
`sampling_rate_hz`.

Access steps, signal handling and commands are in the module README. Everything in this
directory except this file and `metadata.json` is generated and gitignored.
