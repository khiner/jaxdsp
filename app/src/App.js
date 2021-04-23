import { JaxDspClient } from 'jaxdsp-client'

import testSample from './assets/speech-male.wav'

export default function App() {
  return (
    <div>
      <JaxDspClient testSample={testSample} />
    </div>
  )
}
