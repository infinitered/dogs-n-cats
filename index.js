import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_SIZE,
  NUM_CLASSES,
  NUM_DATASET_ELEMENTS,
  NUM_CHANNELS,
  BYTES_PER_UINT8,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS
} from './constants'

const string64_to_float32 = str64 => 
  new Float32Array(new Uint8Array([...atob(str64)].map(c => c.charCodeAt(0))).buffer)

export class DogsNCats {
  constructor() {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  async load() {
    allData = require('dogsNcats.json')
    this.datasetDogs = string64_to_float32(allData.dogs)
    this.datasetCats = string64_to_float32(allData.cats)
  }

  devBatch(batchSize) {
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )
    const startPoint = 0
    const image = this.datasetDogs[0].slice(
      startPoint,
      startPoint + IMAGE_SIZE * NUM_CHANNELS
    )
    batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS)    
    const xs = tf.tensor3d(batchImagesArray, [
      batchSize,
      IMAGE_SIZE,
      NUM_CHANNELS
    ])

    return [xs]
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
        // return this.shuffledTrainIndex // For debugging, no rando
      }
    )
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
      // return this.shuffledTestIndex // For debugging, no rando
    })
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )

    // Create a batchSize of images
    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const startPoint = idx * IMAGE_SIZE * NUM_CHANNELS
      const image = data[0].slice(
        startPoint,
        startPoint + IMAGE_SIZE * NUM_CHANNELS
      )
      batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS)
    }
    const xs = tf.tensor3d(batchImagesArray, [
      batchSize,
      IMAGE_SIZE,
      NUM_CHANNELS
    ])

    return { xs }
  }
}
