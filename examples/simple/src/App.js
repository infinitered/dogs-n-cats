import React, { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs'
import './App.css';
import {DogsNCats} from './dnc'

// const batchSize = 42
const size = 32
const zoom = 2
const gridX = 20
const gridY = 3


function App() {
  const tensorDisplay = useRef(null)
  useEffect(() => {
    const dnc = new DogsNCats()
    dnc.load().then(async () => {
      console.log('Loaded')

      const batchSize = gridX * gridY
      const [batch] = dnc.devBatch(batchSize)
      tensorDisplay.current.width = size*batchSize*zoom
      const tdctx = tensorDisplay.current.getContext('2d')
      // zoom in a little
      tdctx.scale(zoom,zoom)
      let xpos = 0
      let ypos = 0

      tf.unstack(batch).forEach(async tensor => {
        const imageTensor = tensor
          .div(255)
          .reshape([
          size,
          size,
          3
        ])

        // This creation/use must exist inside loop bc
        // foreEach async would fight over shared resources
        const canvas = document.createElement('canvas')
        canvas.width = size
        canvas.height = size    
              
        tensor.print()
        await tf.browser.toPixels(imageTensor, canvas)  
        tdctx.drawImage(canvas, xpos * size, ypos * size, size, size)
        
        xpos++
        tensor.dispose()
        imageTensor.dispose()
      }) 
      
      batch.dispose()
    })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
      <canvas style={styles.canvas} ref={tensorDisplay} />
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
        
      </header>
    </div>
  );
}

const styles = {
  canvas: {
    // maxWidth: '100%',
    alignSelf: 'center'
  }
}

export default App;
