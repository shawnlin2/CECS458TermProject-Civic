import { useState } from 'react'
import Guide from './components/guide'
import './App.css'

function App() {

  return (
    <>
      <div className="container">
        <div className="headerText">
          CivicAI
        </div>
        <div className="guide">
          <Guide></Guide>
        </div>
      </div>
    </>
  )
}

export default App
