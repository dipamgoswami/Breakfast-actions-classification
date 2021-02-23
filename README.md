# Breakfast-actions-classification

Task: Perform video action classification on the Breakfast actions dataset.

Segment Representation: The sub-action segment intervals are generated from the ground truth frame-wise sub-action labels of the videos. The video frames belonging to a single sub-action are stored in the form of npy file and since all the frames in the segment have the same action label, the label of the first frame of the segment is considered as the label of the segment (one-hot label representation). The segments are padded to 6000 length to maintain the same segment length (Max length for segment of 5791 frames).
7075 segments are generated from training split data and 1284 segments are generated from testing data. 5660 segments are used for training while 1415 segments are used for validation. 

Experiments: Bidirectional LSTM are used to incorporate the temporal relations between the segment frames. All the models are trained for 20 epochs.
(A) - 2 Bidirectional LSTM layers followed by Max-pooling and softmax classification layer. Hidden layer dimensions - 400.
(B) - 2 Bidirectional LSTM layers followed by Max-pooling and softmax classification layer. Hidden layer dimensions - 500.
(C) - 3 Bidirectional LSTM layers followed by Max-pooling and softmax classification layer. Hidden layer dimensions - 400.

| Experiment | Validation Accuracy | Test Accuracy |
| (A) | 64.10% | 52.49% |
| (B) | 68.00% | 51.17% |
| (C) | 66.64% | 49.92% |

The keras framework is used for developing the models.
