import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_SIZE,
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_CLASSES,
  NUM_PER_CLASS,
  NUM_DATASET_ELEMENTS,
  NUM_CHANNELS,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS
} from './constants'

const string64_to_float32 = str64 =>
  new Float32Array(
    new Uint8Array([...atob(str64)].map(c => c.charCodeAt(0))).buffer
  )

// Given N datasets, slices the correct amount and combines them
const createSet = (dataSets, quantityTotal, rootOffset = 0) => {

  // requires even total - fix later
  const quantityFromEach = Math.floor(quantityTotal / dataSets.length)
  const scaledRootOffset = rootOffset * IMAGE_SIZE * NUM_CHANNELS
  
  const subset = new Float32Array(
    IMAGE_SIZE * quantityTotal * NUM_CHANNELS
  )
  let localOffset = 0
  dataSets.forEach(dataGroup => {
    const grabSize = IMAGE_SIZE * quantityFromEach * NUM_CHANNELS
    const begin = scaledRootOffset
    const end = grabSize + begin
    subset.set(dataGroup.slice(begin, end), localOffset)
    localOffset += grabSize
  })
  return subset
}

export async function load() {
  if (tf == null) {
    throw new Error(
      `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this module.`
    )
  }

  const dnc = new DogsNCats()
  await dnc.load()
  return dnc
}

export class DogsNCats {
  constructor() {
    this.shuffledDogIndex = 0
    this.shuffledCatIndex = 0
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  async load() {
    const allData = require('./dogsNcats.json')
    // decode JSON data
    this.datasetDogs = string64_to_float32(allData.dogs)
    this.datasetCats = string64_to_float32(allData.cats)
    
    // create data-specific labels/randomization
    this.datasetDogsLabels = new Array(NUM_PER_CLASS).fill(0)
    this.datasetCatsLabels = new Array(NUM_PER_CLASS).fill(1)
    this.shuffledDogIndices = tf.util.createShuffledIndices(
      NUM_PER_CLASS
    )
    this.shuffledCatIndicies = tf.util.createShuffledIndices(
      NUM_PER_CLASS
    )

    // Create training set, labels, indicies with dogs and conCATs :'D
    this.trainSet = createSet(
      [this.datasetDogs, this.datasetCats],
      NUM_TRAIN_ELEMENTS
    )
    this.trainSetLabels = new Array(NUM_TRAIN_ELEMENTS/NUM_CLASSES)
      .fill(0)
      .concat(new Array(NUM_TRAIN_ELEMENTS/NUM_CLASSES).fill(1))
    this.shuffledTrainIndices = tf.util.createShuffledIndices(
      NUM_TRAIN_ELEMENTS
    )
    // Create test set, labels, and indicies
    this.testSet = createSet(
      [this.datasetDogs, this.datasetCats],
      NUM_TEST_ELEMENTS,
      (NUM_TRAIN_ELEMENTS/2)
    )
    this.testSetLabels = new Array(NUM_TEST_ELEMENTS/NUM_CLASSES)
      .fill(0)
      .concat(new Array(NUM_TEST_ELEMENTS/NUM_CLASSES).fill(1))
    this.shuffledTestIndices = tf.util.createShuffledIndices(
      NUM_TEST_ELEMENTS
    )

    // DC.training
    this.training = {
      get: (batchSize = 1) =>
        this.getRandomBatch(
          [this.trainSet, this.trainSetLabels],
          batchSize,
          () => {
            this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.shuffledTrainIndices.length
            return this.shuffledTrainIndices[this.shuffledTrainIndex]
          }
        ),
      getOrdered: (batchSize = 1) => this.getBatch(this.trainSet, batchSize),
      length: this.trainSet.length / IMAGE_SIZE / NUM_CHANNELS
    }

    // DC.test
    this.test = {
      get: (batchSize = 1) =>
        this.getRandomBatch(
          [this.testSet, this.testSetLabels],
          batchSize,
          () => {
            this.shuffledTestIndex =
              (this.shuffledTestIndex + 1) % this.shuffledTestIndices.length
            return this.shuffledTestIndices[this.shuffledTestIndex]
          }
        ),
      getOrdered: (batchSize = 1) => this.getBatch(this.testSet, batchSize),
      length: this.testSet.length / IMAGE_SIZE / NUM_CHANNELS
    }

    // DC.dogs
    this.dogs = {
      get: (batchSize = 1) =>
        this.getRandomBatch(
          [this.datasetDogs, this.datasetDogsLabels],
          batchSize,
          () => {
            this.shuffledDogIndex =
              (this.shuffledDogIndex + 1) % this.shuffledDogIndices.length
            return this.shuffledDogIndices[this.shuffledDogIndex]
          }
        ),
      getOrdered: (batchSize = 1) => this.getBatch(this.datasetDogs, batchSize),
      length: this.datasetDogs.length / IMAGE_SIZE / NUM_CHANNELS
    }

    // DC.cats
    this.cats = {
      get: (batchSize = 1) =>
        this.getRandomBatch(
          [this.datasetCats, this.datasetCatsLabels],
          batchSize,
          () => {
            this.shuffledCatIndex =
              (this.shuffledCatIndex + 1) % this.shuffledCatIndicies.length
            return this.shuffledCatIndicies[this.shuffledCatIndex]
          }
        ),
      getOrdered: (batchSize = 1) => this.getBatch(this.datasetCats, batchSize),
      length: this.datasetCats.length / IMAGE_SIZE / NUM_CHANNELS
    }
  }

  // Prints out tensors to a canvas in grid form
  gridShow(
    dataSet,
    destinationCanvas,
    numX,
    numY,
    config = { scale: 1, grow: true }
  ) {
    const { scale, grow } = config
    if (grow) {
      destinationCanvas.width = IMAGE_WIDTH * numX * scale
      destinationCanvas.height = IMAGE_HEIGHT * numY * scale
    }
    const tdctx = destinationCanvas.getContext('2d')
    // scale in step
    tdctx.scale(scale, scale)
    let xpos = 0
    let ypos = 0

    tf.unstack(dataSet).forEach(async tensor => {
      const imageTensor = tensor
        .div(255)
        .reshape([IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS])

      // This creation/use must exist inside loop bc
      // foreEach async would fight over shared resources
      const canvas = document.createElement('canvas')
      canvas.width = IMAGE_WIDTH
      canvas.height = IMAGE_HEIGHT

      await tf.browser.toPixels(imageTensor, canvas)
      tdctx.drawImage(
        canvas,
        xpos * IMAGE_WIDTH,
        ypos * IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_HEIGHT
      )

      xpos = (xpos + 1) % numX // next column
      if (xpos === 0) ypos++ // next row
      tensor.dispose()
      imageTensor.dispose()
    })
  }

  getRandomBatch(data, batchSize, indexFunc) {
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )
    const batchLabelsArray = new Uint8Array(batchSize)
    // Create a batchSize of images and labels
    for (let i = 0; i < batchSize; i++) {
      const idx = indexFunc()

      const startPoint = idx * IMAGE_SIZE * NUM_CHANNELS
      // one random image by index
      const image = data[0].slice(
        startPoint,
        startPoint + IMAGE_SIZE * NUM_CHANNELS
      )
      batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS)
      // corresponding  label by index
      const label = data[1][idx]
      batchLabelsArray.set([label], i)
    }

    let xs
    // Single 3D tensor for single get
    if (batchSize === 1) {
      xs = tf.tensor3d(
        batchImagesArray,
        [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS],
        'int32'
      )
    } else {
      xs = tf.tensor4d(
        batchImagesArray,
        [batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS],
        'int32'
      )
    }
    const labels = tf.tensor1d(batchLabelsArray)
    return [xs, labels]
  }

  getBatch(dataSet, batchSize) {
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )
    const startPoint = 0
    const image = dataSet.slice(
      startPoint,
      startPoint + IMAGE_SIZE * NUM_CHANNELS * batchSize
    )
    batchImagesArray.set(image, 0)

    let xs
    // Single 3D tensor for single get
    if (batchSize === 1) {
      xs = tf.tensor3d(
        batchImagesArray,
        [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS],
        'int32'
      )
    } else {
      xs = tf.tensor4d(
        batchImagesArray,
        [batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS],
        'int32'
      )
    }

    return xs
  }
}
