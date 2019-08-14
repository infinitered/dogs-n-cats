import React, { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs'
import './App.css';
import {DogsNCats} from './dnc'

const gridX = 20
const gridY = 3

function App() {
  const tensorDogDisplay = useRef(null)
  const tensorCatDisplay = useRef(null)
  useEffect(() => {
    const dnc = new DogsNCats()
    dnc.load().then(async () => {
      console.log('Loaded')

      const batchSize = gridX * gridY
      const [batchDogs] = dnc.dogs.get(batchSize)
      const [batchCats] = dnc.cats.get(batchSize)
      dnc.gridShow(batchDogs, tensorDogDisplay.current, gridX, gridY, {scale:1.5, grow:true})
      dnc.gridShow(batchCats, tensorCatDisplay.current, gridX, gridY, {scale:1.5, grow:true})
      batchDogs.dispose()
      batchCats.dispose()
    })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <canvas style={styles.canvas} ref={tensorDogDisplay} />
        <p>
          Demo of dogs-n-cats NPM library
        </p>
        <img src="dogsncats.jpg" width="100"/>
        <a
          className="App-link"
          href="https://github.com/infinitered/dogs-n-cats"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub Link
        </a>
        <p>1,000 32x32 Dogs and 1,000 32x32 Cats from the <a 
          href="https://www.cs.toronto.edu/~kriz/cifar.html"      
          className="App-link"     
          target="_blank"
          rel="noopener noreferrer">CIFAR-10</a> dataset, all in-memmory JavaScript.</p>
        <canvas style={styles.canvas} ref={tensorCatDisplay} />
      </header>
      <div>
        <img src="dnc_logo.png" class="logo" />
        <h2>Example Usage</h2>
        <script src="https://gist.github.com/GantMan/78189f19457449a09ecc05eb927f686c.js"></script>
      </div>
    </div>
  );
}

const styles = {
  canvas: {
    maxWidth: '100%',
    alignSelf: 'center'
  }
}

export default App;
