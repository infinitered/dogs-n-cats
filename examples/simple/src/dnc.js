import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_SIZE,
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
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
    const allData = require('./dogsNcats.json')
    this.datasetDogs = string64_to_float32(allData.dogs)
    this.datasetCats = string64_to_float32(allData.cats)
    // DC.dogs
    this.dogs = {
      get: (batchSize = 1) => this.getBatch(this.datasetDogs, batchSize)
    }

    // DC.cats
    this.cats = {
      get: (batchSize = 1) => this.getBatch(this.datasetCats, batchSize)
    }
  }

  gridShow(dataSet, destinationCanvas, numX, numY, config={scale:1, grow:true}) {
    const { scale, grow } = config
    if (grow) {
      destinationCanvas.width = IMAGE_WIDTH*numX*scale
      destinationCanvas.height = IMAGE_HEIGHT*numY*scale 
    }
    const tdctx = destinationCanvas.getContext('2d')
    // scale in step
    tdctx.scale(scale,scale)
    let xpos = 0
    let ypos = 0

    tf.unstack(dataSet).forEach(async tensor => {
      const imageTensor = tensor
        .div(255)
        .reshape([
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        NUM_CHANNELS
      ])

      // This creation/use must exist inside loop bc
      // foreEach async would fight over shared resources
      const canvas = document.createElement('canvas')
      canvas.width = IMAGE_WIDTH
      canvas.height = IMAGE_HEIGHT
            
      await tf.browser.toPixels(imageTensor, canvas)  
      tdctx.drawImage(canvas, xpos * IMAGE_WIDTH, ypos * IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
      
      xpos = ((xpos + 1) % numX) // next column
      if (xpos === 0) ypos++ // next row
      tensor.dispose()
      imageTensor.dispose()
    })  
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
