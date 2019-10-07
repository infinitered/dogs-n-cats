export const IMAGE_WIDTH = 32
export const IMAGE_HEIGHT = 32
export const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
export const NUM_CLASSES = 2
export const NUM_DATASET_ELEMENTS = 2000
export const BYTES_PER_UINT8 = 4

// 1, 3, or 4 (Red+Green+Blue+Alpha)
export const NUM_CHANNELS = 3

// Break up dataset into train/test count
export const TRAIN_TEST_RATIO = 4 / 5
export const NUM_TRAIN_ELEMENTS = Math.floor(
  TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS
)
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

// Evenly distributed per class
export const NUM_PER_CLASS = NUM_DATASET_ELEMENTS / NUM_CLASSES