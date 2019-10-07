import React, { useEffect, useRef } from 'react';
//import * as tf from '@tensorflow/tfjs'
import './App.css';
import Gist from 'react-gist'
import * as DogsNCats from "dogs-n-cats";


const gridX = 20
const gridY = 3

function App() {
  const tensorDogDisplay = useRef(null)
  const tensorCatDisplay = useRef(null)
  useEffect(() => {
    DogsNCats.load().then(async dnc => {
      console.log('Loaded')
      console.log('Training size', dnc.training.length)
      console.log('Test size', dnc.test.length)
      console.log('Dogs size', dnc.dogs.length)
      console.log('Cats size', dnc.cats.length)
      // attach to window for debugging access
      window.dnc = dnc
      
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
        <img src="dogsncats.jpg" width="100" alt="dog and cat next to each other" />
        <a
          className="App-link"
          href="https://github.com/infinitered/dogs-n-cats"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>
        <p>1,000 32x32 Dogs and 1,000 32x32 Cats from the <a 
          href="https://www.cs.toronto.edu/~kriz/cifar.html"      
          className="App-link"     
          target="_blank"
          rel="noopener noreferrer">CIFAR-10</a> dataset, all in-memory JavaScript.</p>
        <canvas style={styles.canvas} ref={tensorCatDisplay} />
      </header>
      <div>
        <img src="dnc_logo.png" className="logo" alt="simple dog and cat logo" />
        <h2>Example Usage</h2>
        <Gist id='78189f19457449a09ecc05eb927f686c' file='dog-cat.js' />
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
