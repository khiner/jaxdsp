import React from "react";
// @ts-ignore
import { JaxDspClient } from "jaxdsp-client";
import audioSample from "./assets/speech-male.wav";

function App() {
  return (
    <div className="App">
      <JaxDspClient audioSample={audioSample} />
    </div>
  );
}

export default App;
