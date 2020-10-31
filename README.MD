![Dogs n Cats Logo](_art/dnc_logo.png)

Looking for data for your browser TensorFlow.js project?   Try dogs n cats!

example:
```js
// https://www.cs.toronto.edu/~kriz/cifar.html
import * as DogsNCats from "dogs-n-cats"

// Creates a 1,000 dog and 1,000 cat images in memory
const DC = await DogsNCats.load()

// 4/5 Train/Test 
DC.training.length // 1600
DC.test.length // 400

// Get 1000 random training images
// images is a stacked tensor set of 3D tensors
// labels is a tensor of 0s and 1s (0 is dog, 1 is cat)
const [images, labels] = DC.training.get(1000)

// single random dog 3D tensor
const [dogTensor, dogLabels] = DC.dogs.get()
// 3 random cat tensors stacked (4D), and 1D tensor of answers
const [catTensors, catLabels] = DC.cat.get(3)
// FAST! but non-random
const sameImages = DC.training.getOrdered(1000)

// Draw a dataset to a canvas
const gridX = 20
const gridY = 3
DC.gridShow(catTensors, displayCanvas, gridX, gridY, {scale:1.5, grow:true})

```

# Dogs N Cats TensorFlow.js Data
#### Gathered from CFAR-10 Dataset
https://www.cs.toronto.edu/~kriz/cifar.html

## Code and More
### https://dogs-n-cats.netlify.com/
Demo website located in examples/simple
