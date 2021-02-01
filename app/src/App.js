import { Monitor } from "jaxdsp-client";

import testSample from "./assets/speech-male.wav";

export default function App() {
  return (
    <div>
      <Monitor testSample={testSample} />
    </div>
  );
}
