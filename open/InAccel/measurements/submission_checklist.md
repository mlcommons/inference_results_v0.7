# MLPerf Inference 0.7 Self-Certification Checklist

Name of Certifying Engineer(s): Vangelis Gkiastas, Elias Koromilas

Email of Certifying Engineer(s): egkiastas@inaccel.com, elias@inaccel.com

Name of System(s) Under Test: alveo2.ethz.ch

Does the submission run the same code in accuracy and performance
modes? (check one)
- [X] Yes
- [ ] No

Where is the LoadGen trace stored? (check one)
- [X] Host DRAM
- [ ] Other, please specify:

Are the weights calibrated using data outside of the calibration set?
(check one)
- [ ] Yes
- [X] No

What untimed pre-processing does the submission use? (check all that apply)
- [X] Resize
- [ ] Reorder channels or transpose
- [ ] Pad
- [ ] A single crop
- [ ] Mean subtraction and normalization
- [ ] Convert to whitelisted format
- [ ] No pre-processing
- [ ] Other, please specify:

What numerics does the submission use? (check all that apply)
- [ ] INT4
- [X] INT8
- [ ] INT16
- [ ] INT32
- [ ] UINT8
- [ ] UINT16
- [ ] UINT32
- [ ] FP11
- [ ] FP16
- [ ] BF16
- [ ] FP32
- [ ] Other, please specify:

Which of the following techniques does the submission use? (check all
that apply)
- [X] Wholesale weight replacement
- [ ] Weight supplements
- [ ] Discarding non-zero weight elements
- [ ] Pruning
- [ ] Caching queries
- [ ] Caching responses
- [ ] Caching intermediate computations
- [ ] Modifying weights during the timed portion of an inference run
- [ ] Weight quantization algorithms that are similar in size to the
non-zero weights they produce
- [ ] Hard coding the total number of queries
- [ ] Techniques that boost performance for fixed length experiments but
are inapplicable to long-running services except in the offline
scenario
- [ ] Using knowledge of the LoadGen implementation to predict upcoming
lulls or spikes in the server scenario
- [ ] Treating beams in a beam search differently. For example,
employing different precision for different beams
- [ ] Changing the number of beams per beam search relative to the reference
- [ ] Incorporating explicit statistical information about the performance or accuracy sets
- [ ] Techniques that take advantage of upsampled images.
- [ ] Techniques that only improve performance when there are identical samples in a query.
- [ ] None of the above

Is the submission congruent with all relevant MLPerf rules?
- [X] Yes
- [ ] No

For each SUT, does the submission accurately reflect the real-world
performance of the SUT?
- [X] Yes
- [ ] No
