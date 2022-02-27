import React from "react";
// @ts-ignore
import { JaxDspClient } from "jaxdsp-client";
import testSample from "./assets/speech-male.wav";

function App() {
  return (
    <div className="App">
      <JaxDspClient testSample={testSample} />
    </div>
  );
}

export default App;
