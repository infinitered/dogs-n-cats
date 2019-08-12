import React, { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs'
import './App.css';
import {DogsNCats} from 'dogs-n-cats'

function App() {
  const tensorDisplay = useRef(null)
  useEffect(() => {
    const dnc = new DogsNCats()
    dnc.load().then(async () => {
      console.log('Loaded')
      const [batch] = dnc.devBatch(1)
      tf.unstack(batch).forEach(async tensor => {
        const imageTensor = tensor.div(255).reshape([
          32,
          32,
          3
        ])
        await tf.browser.toPixels(imageTensor, tensorDisplay.current)
    
        tensor.dispose()
        imageTensor.dispose()
      })      
    })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <p>
          Demo of dogs-n-cats NPM library
        </p>
        <a
          className="App-link"
          href="https://github.com/infinitered/dogs-n-cats"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub Link
        </a>
        <canvas style={styles.canvas} ref={tensorDisplay} />
      </header>
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
